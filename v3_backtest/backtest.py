"""Backtest multi-horizonte da previsao do preco do milho (media_pr, R$/saca 60 kg).

Para cada origem t (a cada --step semanas desde --test-start), cada modelo usa
SOMENTE dados ate t e preve t+h para h em --horizons. Familias de modelos:
  BASE     NAIVE (repete o ultimo preco; referencia do MAE relativo = 1, sempre roda)
           SNAIVE (preco da mesma semana do ano anterior)
  PROF     todos os modelos do professor (v2), do jeito dele: preveem o nivel do
           preco com as features dele e prevem varios passos de forma recursiva
           PROF_RIDGE, PROF_ELASTICNET, PROF_RF, PROF_EXTRATREES, PROF_GBR,
           PROF_HISTGBR, PROF_XGBOOST, PROF_LIGHTGBM
  PROFVAR  os mesmos 8 modelos do professor, com as mesmas features e parametros,
           mas aprendendo a VARIACAO do preco em vez do preco (PROFVAR_RF, ...)
  RET      modelos diretos (um por horizonte) que preveem o retorno log
           y[t+h]/y[t] e depois reconstroem o preco: RET_RIDGE, RET_EXTRATREES, RET_GBR
  CHRONOS  modelos pre-treinados da Amazon, zero-shot (sem treino), definidos
           em --chronos-models: CHRONOS_BOLT_SMALL, CHRONOS_2, ...
  COMBO    combinacoes (media/mediana) das previsoes acima, fixadas antes do teste;
           calculadas automaticamente, sem treino (ver COMBOS)

Por padrao, modelos que nao rodaram agora sao mantidos da rodada anterior (mesmas
origens/horizontes), entao da para rodar so um modelo novo. --limpar comeca do zero e
--reavaliar so recalcula metricas, combinacoes e graficos a partir do CSV salvo.

Janela de treino: por padrao cada modelo estuda TODO o passado desde 2001 ate a data da
previsao (janela expansiva). Com --janela-anos N ele estuda so os ultimos N anos antes de
cada previsao (janela deslizante) e os resultados vao para outputs_janelaN/. O Chronos nao
e treinado, entao continua recebendo o historico inteiro.

Janela de entrada: --entrada N limita o que o modelo ENXERGA na hora de cada previsao aos
precos das ultimas N semanas (a semana atual e as N-1 anteriores). Sem a opcao, cada familia
usa as janelas que ja tinha (ate 52 semanas). Resultados em outputs_entradaN/.

Uso:
    .venv/bin/python v3_backtest/backtest.py                          # todos
    .venv/bin/python v3_backtest/backtest.py --models PROF            # so os do professor
    .venv/bin/python v3_backtest/backtest.py --models PROF_RF,RET_GBR,CHRONOS
    .venv/bin/python v3_backtest/backtest.py --horizons 1,4,12 --step 2
    .venv/bin/python v3_backtest/backtest.py --reavaliar             # sem rodar modelos
    .venv/bin/python v3_backtest/backtest.py --janela-anos 10        # treina so com os ultimos 10 anos
    .venv/bin/python v3_backtest/backtest.py --entrada 4             # cada previsao olha so 4 semanas
"""
import argparse
import contextlib
import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
QUANTIS = [0.1, 0.5, 0.9]
# Pracas com muitas lacunas (Pitanga so comeca em 2023) derrubariam quase todo o treino.
MAX_FRACAO_AUSENTE = 0.10
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

MODELOS_PROF = ["RIDGE", "ELASTICNET", "RF", "EXTRATREES", "GBR", "HISTGBR", "XGBOOST", "LIGHTGBM"]
MODELOS_RET = ["RIDGE", "EXTRATREES", "GBR"]
FAMILIAS = ["BASE", "PROF", "PROFVAR", "RET", "CHRONOS", "COMBO"]
NOME_FAMILIA = {"BASE": "Referencias", "PROF": "Professor (nivel, recursivo)",
                "PROFVAR": "Professor prevendo variacao", "RET": "Retorno log (direto)",
                "CHRONOS": "Chronos (zero-shot)", "COMBO": "Combinacoes"}

# Combinacoes fixadas ANTES de ver o teste (escolher as melhores no teste seria trapaca).
# "mediana" sem lista = mediana de todos os modelos que rodaram (menos SNAIVE).
COMBOS = {
    "COMBO_MEDIANA": ("mediana", None),
    "COMBO_NAIVE_RET": ("media", ["NAIVE", "RET_RIDGE"]),
    "COMBO_RET_CHRONOS": ("media", ["RET_RIDGE", "CHRONOS_2"]),
    "COMBO_LINEARES": ("media", ["PROF_RIDGE", "PROF_ELASTICNET", "RET_RIDGE"]),
}

# Paleta categorica validada (ordem fixa) e cor por familia.
CATEGORICAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COR_FAMILIA = {"BASE": "#a3a29c", "PROF": "#2a78d6", "PROFVAR": "#6250d6", "RET": "#eb6834",
               "CHRONOS": "#1baf7a", "COMBO": "#e87ba4"}
COR_NAIVE = "#52514e"
COR_REAL = "#0b0b0b"


def carregar_professor():
    caminho = RAIZ / "v2_modelo_professor" / "02_modelo_previsoes_milho.py"
    spec = importlib.util.spec_from_file_location("modelo_professor", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nome_chronos(model_id):
    return "CHRONOS_" + model_id.split("/")[-1].replace("chronos-", "").replace("-", "_").upper()


def familia(modelo):
    return "BASE" if modelo in ("NAIVE", "SNAIVE") else modelo.split("_")[0]


def catalogo(chronos_ids):
    """Todos os nomes de modelo, na ordem fixa usada nas tabelas e cores."""
    return (["NAIVE", "SNAIVE"] + [f"PROF_{m}" for m in MODELOS_PROF] + [f"PROFVAR_{m}" for m in MODELOS_PROF]
            + [f"RET_{m}" for m in MODELOS_RET] + [nome_chronos(c) for c in chronos_ids] + list(COMBOS))


def selecionar_modelos(texto, todos):
    pedidos = [p.strip().upper() for p in texto.split(",") if p.strip()]
    if not pedidos or "ALL" in pedidos:
        return list(todos)
    escolhidos = {"NAIVE"}  # referencia do MAE relativo
    for p in pedidos:
        if p in FAMILIAS:
            escolhidos.update(m for m in todos if familia(m) == p)
        elif p in todos:
            escolhidos.add(p)
        else:
            raise SystemExit(f"Modelo desconhecido: {p}. Opcoes: ALL, {', '.join(FAMILIAS)}, {', '.join(todos)}")
    return [m for m in todos if m in escolhidos]


def cor(modelo, todos):
    """Cor fixa do modelo dentro da sua familia (nao muda com o filtro --models)."""
    if modelo == "NAIVE":
        return COR_NAIVE
    irmaos = [m for m in todos if familia(m) == familia(modelo)]
    return CATEGORICAS[irmaos.index(modelo) % len(CATEGORICAS)]


# ============================================================
# VARIACAO, COMBINACOES, SIGNIFICANCIA E MESCLA DE RESULTADOS
# ============================================================

class NivelViaVariacao:
    """Faz um modelo do professor aprender a variacao log do preco, log(y[t] / y[t-1]),
    em vez do preco. Por fora ele continua recebendo as mesmas features e devolvendo o
    preco (lag_1 * exp(variacao prevista)), entao a previsao recursiva do professor
    funciona sem mudanca.

    A variacao prevista e limitada a maior variacao semanal vista no treino: na previsao
    recursiva uma variacao absurda vira entrada da semana seguinte e o preco "explode"
    (chega a infinito), principalmente com pouco treino (--janela-anos) ou pouca entrada."""

    def __init__(self, modelo, idx_lag1):
        self.modelo, self.idx = modelo, idx_lag1

    def fit(self, X, y):
        var = np.log(y / X[:, self.idx])
        self.limite = float(np.nanmax(np.abs(var)))
        self.modelo.fit(X, var)
        return self

    def predict(self, X):
        var = np.clip(self.modelo.predict(X), -self.limite, self.limite)
        return X[:, self.idx] * np.exp(var)


def ler_previsoes(arquivo):
    return pd.read_csv(arquivo, parse_dates=["data_origem", "data_alvo"])


def mesclar_com_anterior(prev, arquivo, chaves):
    """Junta as previsoes novas com as da rodada anterior, mantendo os modelos que nao
    rodaram agora. So mescla se o periodo de teste for o mesmo."""
    arquivo = Path(arquivo)
    if not arquivo.exists():
        return prev
    ant = ler_previsoes(arquivo)
    chaves_de = lambda df: set(map(tuple, df.loc[df.modelo == "NAIVE", chaves].to_numpy().tolist()))
    if chaves_de(ant) != chaves_de(prev):
        print("Rodada anterior tem outro periodo/horizonte: resultados antigos nao foram mantidos.")
        return prev
    manter = ant[~ant.modelo.isin(set(prev.modelo)) & (ant.modelo.map(familia) != "COMBO")]
    if len(manter):
        print(f"Mantidos da rodada anterior: {', '.join(sorted(manter.modelo.unique()))}")
    return pd.concat([prev, manter], ignore_index=True)


def adicionar_combinacoes(prev, chaves):
    """Acrescenta as previsoes COMBO_* (ver COMBOS) a partir das previsoes dos outros modelos."""
    prev = prev[prev.modelo.map(familia) != "COMBO"].reset_index(drop=True)
    piv = prev.pivot_table(index=chaves, columns="modelo", values="pred", aggfunc="first")
    base = (prev[prev.modelo == "NAIVE"].set_index(chaves)
            .drop(columns=["modelo", "pred", "q10", "q90"], errors="ignore"))
    novas = []
    for nome, (tipo, comps) in COMBOS.items():
        if comps is None:
            comps = [c for c in piv.columns if c != "SNAIVE"]
        if len(comps) < 2 or not set(comps) <= set(piv.columns):
            continue
        p = piv[comps].median(axis=1, skipna=False) if tipo == "mediana" else piv[comps].mean(axis=1, skipna=False)
        novas.append(base.assign(modelo=nome, pred=p.reindex(base.index)).dropna(subset=["pred"]).reset_index())
    return pd.concat([prev] + novas, ignore_index=True) if novas else prev


def diebold_mariano(d, lag):
    """p-valor bilateral do teste de Diebold-Mariano, com a correcao de Harvey, Leybourne e
    Newbold para amostras pequenas. d = |erro do modelo| - |erro do ingenuo|, em ordem de
    tempo. H0: os dois tem o mesmo erro medio. lag = quantas previsoes vizinhas
    compartilham erro (h - 1 quando as previsoes de h passos se sobrepoem)."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan
    dc = d - d.mean()
    lag = int(min(max(lag, 0), n - 2))
    var = (dc @ dc + 2 * sum(dc[k:] @ dc[:-k] for k in range(1, lag + 1))) / n / n
    if var <= 0:
        return np.nan
    h = lag + 1
    correcao = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t = d.mean() / np.sqrt(var) * correcao
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def comparacao_naive(rel, p):
    if not np.isfinite(p):
        return ""
    if p < 0.05:
        return "melhor (p<0,05)" if rel < 1 else "pior (p<0,05)"
    return "sem diferenca"


def grafico_prof_nivel_vs_variacao(metricas, out, arquivo="06_professor_nivel_vs_variacao.png"):
    """Mesmos modelos do professor: prevendo o preco (PROF_) x prevendo a variacao (PROFVAR_)."""
    m = metricas[metricas.familia.isin(["PROF", "PROFVAR"])]
    nomes = [n for n in MODELOS_PROF if {f"PROF_{n}", f"PROFVAR_{n}"} <= set(m.modelo)]
    if not nomes:
        return
    from matplotlib.ticker import FuncFormatter
    hs = sorted(m.h.unique()) if "h" in m else [None]
    fig, axes = plt.subplots(1, len(hs), figsize=(max(9, 4.4 * len(hs) + 1.8), 0.55 * len(nomes) + 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    yy = np.arange(len(nomes))
    todos_vals = m.MAE_rel_naive[np.isfinite(m.MAE_rel_naive)]
    lim = (min(0.7, todos_vals.min() * 0.85), max(1.5, todos_vals.max() * 1.6))
    for ax, h in zip(axes, hs):
        g = (m if h is None else m[m.h == h]).set_index("modelo").MAE_rel_naive
        for desloc, fam, rotulo in ((-0.2, "PROF", "preve o preco (original)"),
                                    (0.2, "PROFVAR", "preve a variacao")):
            vals = [g.get(f"{fam}_{n}", np.nan) for n in nomes]
            ax.barh(yy + desloc, vals, height=0.38, color=COR_FAMILIA[fam], label=rotulo,
                    edgecolor="white", linewidth=1.5)
            for yi, v in zip(yy + desloc, vals):
                if np.isfinite(v):
                    ax.text(v, yi, f" {v:.2f}", va="center", fontsize=7.5, color="#52514e")
        ax.axvline(1, color=COR_REAL, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        ax.set_xticks([t for t in (0.5, 1, 2, 5, 10, 20) if lim[0] <= t <= lim[1]])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_yticks(yy)
        ax.set_yticklabels(nomes)
        ax.invert_yaxis()
        estilo(ax, "" if h is None else f"h = {h} semana(s)", "")
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Modelos do professor: prevendo o preco x prevendo a variacao\n"
                 "Erro relativo ao ingenuo (escala log; < 1 = melhor)", x=0.01, ha="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / arquivo, dpi=160)
    plt.close(fig)


def inicio_janela(datas, t, anos):
    """Primeira linha que pode entrar no treino de uma previsao feita na linha t.
    anos=None: janela expansiva (desde o inicio da serie). anos=N: janela deslizante, so as
    semanas dos ultimos N anos antes de datas[t]."""
    if not anos:
        return 0
    return int(np.searchsorted(datas, datas.iloc[t] - pd.DateOffset(years=anos)))


def sufixo_saida(janela=None, entrada=None):
    """Nome da pasta de resultados de cada variante: outputs, outputs_janela10, outputs_entrada4..."""
    return (f"_janela{janela}" if janela else "") + (f"_entrada{entrada}" if entrada else "")


def colunas_prof_entrada(cols, entrada):
    """Das features do professor, so as que usam os precos das ultimas `entrada` semanas.
    (No script dele a linha j preve y[j]; a "semana atual" e j-1, entao lag_k usa y[j-k] e
    roll_*_w usa y[j-1]...y[j-w].) Calendario (t, ano, sin/cos) e pracas na semana atual ficam."""
    if not entrada:
        return cols

    def cabe(c):
        if c.startswith("lag_"):
            return int(c.split("_")[1]) <= entrada
        if c.startswith("roll_"):
            return int(c.rsplit("_", 1)[1]) <= entrada
        if c == "diff_1":
            return entrada >= 2
        if c == "diff_52":
            return entrada >= 53
        return True
    return [c for c in cols if cabe(c)]


def contexto(y, t, entrada):
    """Historico que o Chronos recebe numa previsao feita na linha t."""
    return y.iloc[(max(0, t + 1 - entrada) if entrada else 0): t + 1].to_numpy()


# ============================================================
# DADOS
# ============================================================

def carregar_dados(path, target):
    df = pd.read_csv(path, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    pracas = [c for c in df.columns if c not in ("data", "media_deral", "media_pr")]
    boas = [c for c in pracas if df[c].isna().mean() <= MAX_FRACAO_AUSENTE]
    removidas = sorted(set(pracas) - set(boas))
    print(f"Linhas: {len(df)} | {df.data.min().date()} a {df.data.max().date()}")
    print(f"Pracas usadas: {len(boas)} | removidas por lacunas: {removidas}")
    y_original = df[target].astype(float)
    # Preenchimento so com o passado (para lags); rotulos ausentes nunca entram no treino/teste.
    y = y_original.ffill()
    pracas_ff = df[boas].ffill()
    return df.data, y, y_original, pracas_ff


def features_diretas(y, pracas, datas, entrada=None):
    """Features no instante t usando apenas informacao ate t.
    entrada=N: so features calculadas com os precos das ultimas N semanas (N precos = N-1
    variacoes); nesse caso entram todas as variacoes semana a semana da janela."""
    logy = np.log(y)
    r = logy.diff()
    f = pd.DataFrame(index=y.index)
    n_var = entrada - 1 if entrada else 52  # quantas variacoes semanais cabem na janela
    for k in (range(n_var) if entrada else (0, 1, 2, 3, 5, 11)):
        f[f"ret_lag{k}"] = r.shift(k)
    for w in (4, 12, 26, 52):
        if w <= n_var:
            f[f"ret_acum_{w}"] = logy - logy.shift(w)
    for w in (4, 12, 26):
        if w <= n_var:
            f[f"vol_{w}"] = r.rolling(w).std()
    for w in (12, 26, 52):
        if w <= n_var + 1:
            f[f"dist_media_{w}"] = logy - np.log(y.rolling(w).mean())
    semana = datas.dt.isocalendar().week.astype(float)
    f["sin_sem"] = np.sin(2 * np.pi * semana / 52.18)
    f["cos_sem"] = np.cos(2 * np.pi * semana / 52.18)
    # Dispersao entre pracas: mercado "descolado" tende a convergir.
    f["dispersao_pracas"] = pracas.std(axis=1) / pracas.mean(axis=1)
    f["amplitude_pracas"] = np.log(pracas.max(axis=1) / pracas.min(axis=1))
    return f


def modelos_ret(nomes):
    todos = {
        "RIDGE": lambda: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "EXTRATREES": lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                                  random_state=RANDOM_STATE, n_jobs=-1),
        "GBR": lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                                 subsample=0.8, random_state=RANDOM_STATE),
    }
    return {n: todos[n]() for n in nomes}


# ============================================================
# BACKTEST
# ============================================================

def backtest_baselines(y, origens, horizons, usar_snaive):
    linhas = []
    for t in origens:
        for h in horizons:
            linhas.append({"origem": t, "h": h, "modelo": "NAIVE", "pred": y.iloc[t]})
            if usar_snaive and t + h - 52 >= 0:
                linhas.append({"origem": t, "h": h, "modelo": "SNAIVE", "pred": y.iloc[t + h - 52]})
    return linhas


def backtest_ret(nomes, y, y_original, feats, origens, horizons, datas=None, janela=None):
    linhas = []
    logy = np.log(y)
    for h in horizons:
        alvo = logy.shift(-h) - logy  # retorno log ate t+h
        rotulo_ok = y_original.shift(-h).notna()
        for t in origens:
            # rotulo de i conhecido em t se i + h <= t; a janela corta o comeco
            idx = np.arange(inicio_janela(datas, t, janela), t - h + 1)
            treino = feats.iloc[idx].assign(_alvo=alvo.iloc[idx])
            treino = treino[rotulo_ok.iloc[idx].values].dropna()
            x_t = feats.iloc[[t]]
            if x_t.isna().any().any() or len(treino) < 52:
                continue
            X, Y = treino.drop(columns="_alvo").to_numpy(), treino._alvo.to_numpy()
            for nome, modelo in modelos_ret(nomes).items():
                modelo.fit(X, Y)
                ret = float(modelo.predict(x_t.to_numpy())[0])
                linhas.append({"origem": t, "h": h, "modelo": f"RET_{nome}", "pred": y.iloc[t] * np.exp(ret)})
        print(f"  RET h={h} ok")
    return linhas


def backtest_professor(nomes, datas, y, y_original, pracas, origens, horizons, variacao=False, janela=None,
                       entrada=None):
    """Modelos do professor (mesmas features, hiperparametros e previsao recursiva),
    sem as pracas esparsas. variacao=True: aprendem a variacao do preco (PROFVAR_)."""
    prefixo = "PROFVAR" if variacao else "PROF"
    prof = carregar_professor()
    with contextlib.redirect_stdout(io.StringIO()):  # avisos repetidos do build_models
        disponiveis = set(prof.build_models(nomes))
    faltando = [n for n in nomes if n not in disponiveis]
    if faltando:
        print(f"  Indisponiveis (instale xgboost/lightgbm): {faltando}")
    nomes = [n for n in nomes if n in disponiveis]

    hist = pd.DataFrame({"timestamp": datas, "target": y})
    for i, c in enumerate(pracas.columns, start=1):
        hist[f"aux_{i}"] = pracas[c].values
    feats = prof.add_time_features(hist)
    feats.loc[y_original.isna(), "target"] = np.nan
    cols = colunas_prof_entrada(prof.get_feature_columns(feats), entrada)
    linhas = []
    for nome in nomes:
        t0 = time.perf_counter()
        for t in origens:
            treino = feats.iloc[inicio_janela(datas, t, janela): t + 1].dropna(subset=cols + ["target"])
            with contextlib.redirect_stdout(io.StringIO()):
                modelo = prof.build_models([nome])[nome]
            if variacao:
                modelo = NivelViaVariacao(modelo, cols.index("lag_1"))
            modelo.fit(treino[cols].to_numpy(float), treino.target.to_numpy(float))
            fut = prof.recursive_forecast(modelo, hist.iloc[: t + 1], cols, max(horizons), "W-FRI")
            for h in horizons:
                linhas.append({"origem": t, "h": h, "modelo": f"{prefixo}_{nome}",
                               "pred": fut.previsao_target.iloc[h - 1]})
        print(f"  {prefixo}_{nome} ok [{time.perf_counter() - t0:.0f}s]")
    return linhas


def carregar_chronos(model_id):
    import torch
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)


def prever_chronos(pipe, contextos, passos):
    """Retorna array (n_series, passos, len(QUANTIS))."""
    import torch
    q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in contextos],
                                  prediction_length=passos, quantile_levels=QUANTIS)
    if isinstance(q, list):  # Chronos-2: lista de (n_variaveis, passos, quantis)
        q = torch.stack([qi[0] for qi in q])
    return q.float().numpy()


def rodar_chronos(model_id, contextos, passos, serie_completa, passos_futuros):
    """Roda em processo separado: o torch e o XGBoost/LightGBM trazem cada um sua libomp,
    e no macOS as duas no mesmo processo travam ou derrubam o Python."""
    pipe = carregar_chronos(model_id)
    return prever_chronos(pipe, contextos, passos), prever_chronos(pipe, [serie_completa], passos_futuros)[0]


def backtest_chronos(model_id, nome, y, origens, horizons, passos_futuros, entrada=None):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    contextos = [contexto(y, t, entrada) for t in origens]
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        q, q_fut = ex.submit(rodar_chronos, model_id, contextos, max(horizons),
                             contexto(y, len(y) - 1, entrada), passos_futuros).result()
    linhas = []
    for i, t in enumerate(origens):
        for h in horizons:
            linhas.append({"origem": t, "h": h, "modelo": nome, "pred": q[i, h - 1, 1],
                           "q10": q[i, h - 1, 0], "q90": q[i, h - 1, 2]})
    return linhas, q_fut


# ============================================================
# METRICAS
# ============================================================

def calcular_metricas(prev, step=1):
    """step = semanas entre origens; define quantas previsoes vizinhas se sobrepoem no teste DM."""
    linhas = []
    naive = prev[prev.modelo == "NAIVE"].set_index(["origem", "h"]).pred
    for (modelo, h), g in prev.groupby(["modelo", "h"]):
        g = g.dropna(subset=["real"]).sort_values("origem")
        erro = g.real - g.pred
        mae = erro.abs().mean()
        chaves = list(zip(g.origem, g.h))
        mae_naive = np.abs(g.real.values - naive.loc[chaves].values).mean()
        variacao_real = g.real - g.y_origem
        variacao_prev = g.pred - g.y_origem
        mov = variacao_real != 0
        direcao = np.nan if modelo == "NAIVE" else 100 * np.mean(
            np.sign(variacao_real[mov]) == np.sign(variacao_prev[mov]))
        mse = (erro ** 2).mean()
        mse_naive = ((g.real.values - naive.loc[chaves].values) ** 2).mean()
        linha = {"modelo": modelo, "familia": familia(modelo), "h": h, "n": len(g), "MAE": mae,
                 "MSE": mse, "RMSE": np.sqrt(mse),
                 "MAPE_percent": 100 * (erro.abs() / g.real).mean(),
                 "MAE_rel_naive": mae / mae_naive,
                 "RMSE_rel_naive": np.sqrt(mse / mse_naive),
                 "acerto_direcao_percent": direcao}
        d = erro.abs().values - np.abs(g.real.values - naive.loc[chaves].values)
        linha["dm_pvalor"] = diebold_mariano(d, int(np.ceil(h / step)) - 1)
        linha["vs_naive"] = comparacao_naive(linha["MAE_rel_naive"], linha["dm_pvalor"])
        if "q10" in g and g.q10.notna().any():
            linha["cobertura_80_percent"] = 100 * np.mean((g.real >= g.q10) & (g.real <= g.q90))
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values(["h", "MAE"]).reset_index(drop=True)


def tabela_ranking(metricas, ordem):
    """Modelos x horizontes com o MAE relativo ao ingenuo, ordenado pela media."""
    tab = metricas.pivot(index="modelo", columns="h", values="MAE_rel_naive")
    tab.columns = [f"h{h}" for h in tab.columns]
    tab["media"] = tab.mean(axis=1)
    tab = tab.reindex([m for m in ordem if m in tab.index])
    return tab.sort_values("media")


# ============================================================
# GRAFICOS
# ============================================================

def estilo(ax, titulo, ylabel, xlabel=None):
    ax.set_title(titulo, loc="left", fontsize=11, color=COR_REAL)
    ax.set_ylabel(ylabel, color="#52514e")
    if xlabel:
        ax.set_xlabel(xlabel, color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")


def legenda_familias(fig, familias):
    from matplotlib.patches import Patch
    fig.legend([Patch(color=COR_FAMILIA[f]) for f in familias], [NOME_FAMILIA[f] for f in familias],
               loc="lower center", ncol=min(len(familias), 3), frameon=False, fontsize=9)


def grafico_erro_horizonte(metricas, todos, out):
    """Um painel por familia, sempre com o ingenuo tracejado como referencia."""
    familias = [f for f in FAMILIAS if f != "BASE" and (metricas.familia == f).any()]
    if not familias:
        return
    hs = sorted(metricas.h.unique())
    fig, axes = plt.subplots(1, len(familias), figsize=(5.5 * len(familias), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    naive = metricas[metricas.modelo == "NAIVE"].sort_values("h")
    for ax, fam in zip(axes, familias):
        ax.plot(naive.h, naive.MAPE_percent, "--", color=COR_NAIVE, linewidth=2, label="NAIVE")
        for m in [m for m in todos if familia(m) == fam and m in set(metricas.modelo)]:
            g = metricas[metricas.modelo == m].sort_values("h")
            ax.plot(g.h, g.MAPE_percent, color=cor(m, todos), linewidth=2, marker="o", markersize=6,
                    label=m.split("_", 1)[1] if fam != "CHRONOS" else m)
        ax.set_xticks(hs)
        estilo(ax, NOME_FAMILIA[fam], "MAPE (%)" if ax is axes[0] else "", "Horizonte (semanas a frente)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Erro medio (MAPE) por horizonte de previsao", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "01_mape_por_horizonte.png", dpi=160)
    plt.close(fig)


def grafico_barras_por_horizonte(metricas, todos, coluna, referencia, titulo, arquivo, out, fmt="{:.2f}",
                                 estrela=False):
    """Barras horizontais: um painel por horizonte, todos os modelos, cor = familia."""
    modelos = [m for m in todos if m in set(metricas.modelo) and m not in ("NAIVE", "SNAIVE")]
    if not modelos:
        return
    hs = sorted(metricas.h.unique())
    altura = max(4.5, 0.38 * len(modelos) + 1.8)
    fig, axes = plt.subplots(1, len(hs), figsize=(3.8 * len(hs) + 1.5, altura), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs):
        g = metricas[metricas.h == h].set_index("modelo")
        vals = [g[coluna].get(m, np.nan) for m in modelos]
        ax.barh(modelos, vals, color=[COR_FAMILIA[familia(m)] for m in modelos],
                height=0.72, edgecolor="white", linewidth=2)
        ax.axvline(referencia, color=COR_REAL, linewidth=1)
        validos = [v for v in vals if np.isfinite(v)]
        ax.set_xlim(0, max(validos + [referencia]) * 1.15)
        sig = [estrela and g.get("dm_pvalor", pd.Series(dtype=float)).get(m, np.nan) < 0.05 for m in modelos]
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v, i, " " + fmt.format(v) + ("*" if sig[i] else ""), va="center", fontsize=8,
                        color="#52514e")
        estilo(ax, f"h = {h} semana(s)", "")
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12)
    legenda_familias(fig, [f for f in FAMILIAS if f != "BASE" and any(familia(m) == f for m in modelos)])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / arquivo, dpi=160)
    plt.close(fig)


def grafico_real_vs_previsto(prev, metricas, datas, y_original, h, out):
    """Real x NAIVE x melhor modelo de cada familia nesse horizonte."""
    g = prev[prev.h == h]
    if g.empty:
        return
    mh = metricas[metricas.h == h]
    destaque = ["NAIVE"] + [mh[mh.familia == f].iloc[0].modelo
                            for f in FAMILIAS if f != "BASE" and (mh.familia == f).any()]
    inicio = datas.iloc[g.origem.min()] - pd.Timedelta(weeks=8)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    mask = datas >= inicio
    ax.plot(datas[mask], y_original[mask], color=COR_REAL, linewidth=2, label="Real")
    for m in destaque:
        gm = g[g.modelo == m].sort_values("alvo")
        rel = mh.set_index("modelo").MAE_rel_naive.get(m, np.nan)
        ax.plot(datas.iloc[gm.alvo], gm.pred, linewidth=1.5, marker="o", markersize=4,
                color=COR_NAIVE if m == "NAIVE" else COR_FAMILIA[familia(m)],
                linestyle="--" if m == "NAIVE" else "-",
                label=m if m == "NAIVE" else f"{m} (MAE rel. {rel:.2f})")
    estilo(ax, f"Real x previsto com {h} semana(s) de antecedencia: melhor modelo de cada familia",
           "R$/saca 60 kg")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(out / f"03_real_vs_previsto_h{h:02d}.png", dpi=160)
    plt.close(fig)


def grafico_futuro_chronos(futuros, todos, datas, y_original, out):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    mask = datas >= datas.iloc[-1] - pd.Timedelta(weeks=156)
    ax.plot(datas[mask], y_original[mask], color=COR_REAL, linewidth=2, label="Historico")
    for nome, f in futuros.items():
        c = cor(nome, todos)
        ax.fill_between(f.data, f.q10, f.q90, color=c, alpha=0.18, linewidth=0)
        ax.plot(f.data, f.q50, color=c, linewidth=2, label=f"{nome} (mediana, faixa 10-90%)")
    estilo(ax, "Previsao futura zero-shot com faixa de incerteza", "R$/saca 60 kg")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "05_chronos_previsao_futura.png", dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", default=RAIZ / "dados" / "milho_semanal.csv")
    parser.add_argument("--target", default="media_pr")
    parser.add_argument("--output-dir", default=None,
                        help="Padrao: v3_backtest/outputs[_janelaN][_entradaN]")
    parser.add_argument("--models", default="ALL",
                        help="ALL, familias (BASE, PROF, PROFVAR, RET, CHRONOS) e/ou nomes, separados por virgula. "
                             "NAIVE sempre roda.")
    parser.add_argument("--horizons", default="1,4,12,26")
    parser.add_argument("--test-start", default="2023-01-01", help="Primeira origem do backtest.")
    parser.add_argument("--step", type=int, default=4, help="Semanas entre origens.")
    parser.add_argument("--future-steps", type=int, default=26)
    parser.add_argument("--chronos-models", default="amazon/chronos-bolt-small,amazon/chronos-2",
                        help="IDs do Hugging Face que formam a familia CHRONOS.")
    parser.add_argument("--janela-anos", type=int, default=None,
                        help="Janela deslizante: treina so com os ultimos N anos antes de cada previsao.")
    parser.add_argument("--entrada", type=int, default=None,
                        help="Janela de entrada: cada previsao so ve os precos das ultimas N semanas.")
    parser.add_argument("--reavaliar", action="store_true",
                        help="Nao roda modelos: recalcula metricas, combinacoes e graficos do CSV salvo.")
    parser.add_argument("--limpar", action="store_true",
                        help="Descarta os resultados anteriores (por padrao, sao mantidos os modelos que nao rodaram).")
    args = parser.parse_args()

    chronos_ids = [m.strip() for m in args.chronos_models.split(",") if m.strip()]
    todos = catalogo(chronos_ids)
    escolhidos = selecionar_modelos(args.models, todos)
    print(f"Modelos: {', '.join(escolhidos)}")

    if args.entrada is not None and args.entrada < 1:
        parser.error("--entrada deve ser pelo menos 1")
    out = Path(args.output_dir) if args.output_dir else AQUI / f"outputs{sufixo_saida(args.janela_anos, args.entrada)}"
    out.mkdir(parents=True, exist_ok=True)
    horizons = sorted(int(h) for h in args.horizons.split(","))
    datas, y, y_original, pracas = carregar_dados(args.data_file, args.target)
    n = len(y)
    inicio = int(np.searchsorted(datas, pd.Timestamp(args.test_start)))
    origens = list(range(inicio, n - min(horizons), args.step))
    print(f"Origens: {len(origens)} ({datas.iloc[origens[0]].date()} a {datas.iloc[origens[-1]].date()})")
    print("Janela de treino: " + (f"deslizante, ultimos {args.janela_anos} anos antes de cada previsao"
                                  if args.janela_anos else "expansiva, todo o historico desde o inicio"))
    print("Janela de entrada: " + (f"precos das ultimas {args.entrada} semanas" if args.entrada
                                   else "padrao de cada modelo (ate 52 semanas)"))

    def da_familia(f):
        return [m.split("_", 1)[1] for m in escolhidos if familia(m) == f]

    t0 = time.perf_counter()
    arquivo = out / "previsoes_backtest.csv"
    futuros = {}
    if args.reavaliar:
        prev = ler_previsoes(arquivo)
        for f in sorted(out.glob("previsao_futura_*.csv")):
            futuros[f.stem.replace("previsao_futura_", "")] = pd.read_csv(f, parse_dates=["data"])
        print(f"Reavaliando {arquivo} ({prev.modelo.nunique()} modelos), sem rodar nada.")
    else:
        prev = rodar_modelos(args, escolhidos, chronos_ids, datas, y, y_original, pracas, origens, horizons,
                             futuros, t0)
        if not args.limpar:
            prev = mesclar_com_anterior(prev, arquivo, ["origem", "h"])
    prev = adicionar_combinacoes(prev, ["origem", "h"])

    metricas = calcular_metricas(prev, args.step)
    ranking = tabela_ranking(metricas, todos)
    prev.to_csv(arquivo, index=False)
    metricas.to_csv(out / "metricas_por_horizonte.csv", index=False)
    ranking.to_csv(out / "ranking_mae_relativo.csv", float_format="%.3f")
    for nome, f in futuros.items():
        f.to_csv(out / f"previsao_futura_{nome}.csv", index=False)

    grafico_erro_horizonte(metricas, todos, out)
    grafico_barras_por_horizonte(metricas, todos, "MAE_rel_naive", 1,
                                 "MAE relativo ao ingenuo (< 1 = melhor que repetir o ultimo preco; "
                                 "* = diferenca significativa, p < 0,05)",
                                 "02_mae_relativo_ingenuo.png", out, estrela=True)
    for h in sorted(prev.h.unique()):
        grafico_real_vs_previsto(prev, metricas, datas, y_original, h, out)
    grafico_barras_por_horizonte(metricas, todos, "acerto_direcao_percent", 50,
                                 "Acerto da direcao: o preco sobe ou cai ate t+h? (linha = 50%, moeda)",
                                 "04_acerto_direcao.png", out, fmt="{:.0f}%")
    if futuros:
        grafico_futuro_chronos(futuros, todos, datas, y_original, out)
    grafico_prof_nivel_vs_variacao(metricas, out)

    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 160):
        print("\nMAE relativo ao ingenuo (< 1 = melhor), ordenado pela media dos horizontes:")
        print(ranking.to_string())
        sig = metricas[metricas.vs_naive.str.startswith("melhor")]
        print("\nSignificativamente melhores que o ingenuo (Diebold-Mariano, p < 0,05):")
        print(sig[["modelo", "h", "MAE_rel_naive", "dm_pvalor"]].to_string(index=False) if len(sig) else "  nenhum")
    print(f"\nTempo total: {time.perf_counter() - t0:.0f}s | Saidas: {out.resolve()}")


def rodar_modelos(args, escolhidos, chronos_ids, datas, y, y_original, pracas, origens, horizons, futuros, t0):
    n = len(y)

    def da_familia(f):
        return [m.split("_", 1)[1] for m in escolhidos if familia(m) == f]

    linhas = backtest_baselines(y, origens, horizons, "SNAIVE" in escolhidos)
    if da_familia("RET"):
        print("Modelos de retorno (diretos)...")
        linhas += backtest_ret(da_familia("RET"), y, y_original, features_diretas(y, pracas, datas, args.entrada),
                               origens, horizons, datas, args.janela_anos)
    if da_familia("PROF"):
        print(f"Modelos do professor (recursivos)... [{time.perf_counter() - t0:.0f}s]")
        linhas += backtest_professor(da_familia("PROF"), datas, y, y_original, pracas, origens, horizons,
                                     janela=args.janela_anos, entrada=args.entrada)
    if da_familia("PROFVAR"):
        print(f"Modelos do professor prevendo variacao... [{time.perf_counter() - t0:.0f}s]")
        linhas += backtest_professor(da_familia("PROFVAR"), datas, y, y_original, pracas, origens, horizons,
                                     variacao=True, janela=args.janela_anos, entrada=args.entrada)

    for model_id in chronos_ids:
        nome = nome_chronos(model_id)
        if nome not in escolhidos:
            continue
        print(f"{nome} ({model_id})... [{time.perf_counter() - t0:.0f}s]")
        novas, q = backtest_chronos(model_id, nome, y, origens, horizons, args.future_steps, args.entrada)
        linhas += novas
        futuros[nome] = pd.DataFrame({
            "data": datas.iloc[-1] + pd.to_timedelta(7 * np.arange(1, args.future_steps + 1), unit="D"),
            "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]})

    prev = pd.DataFrame(linhas)
    prev["alvo"] = prev.origem + prev.h
    prev = prev[prev.alvo < n].copy()
    prev["data_origem"] = datas.iloc[prev.origem].values
    prev["data_alvo"] = datas.iloc[prev.alvo].values
    prev["y_origem"] = y.iloc[prev.origem].values
    prev["real"] = y_original.iloc[prev.alvo].values
    return prev


if __name__ == "__main__":
    main()

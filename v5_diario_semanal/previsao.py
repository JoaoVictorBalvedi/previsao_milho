"""Holdout 80/20 com escolha do produto (milho ou soja) e da frequencia (diaria ou semanal).

Faz tudo o que a v4 (v4_holdout/holdout.py) faz: modos passo e cego, --models, --treino,
--janela-anos, --entrada, --reavaliar, --limpar, combinacoes, Diebold-Mariano e graficos.
As novidades sao as opcoes --produto e --frequencia:

  --produto milho (padrao)  SIMA/DERAL, media das pracas do Parana (media_pr), 2001-2025;
                            as pracas entram como variaveis auxiliares (dados_milho/milho_*.csv).
  --produto soja            indicador CEPEA/ESALQ Parana a vista em R$ (preco_rs), 2005-2026;
                            serie unica, sem pracas (dados_soja/soja_*.csv, gerados por
                            dados_soja/extrair_soja.py a partir da planilha do CEPEA).

  --frequencia semanal (padrao)  media da semana terminando na sexta. --horizonte e
                    --entrada em semanas. No milho, da o mesmo resultado da v4.
  --frequencia diaria  um passo = um dia util com cotacao (~240 por ano no milho, ~250 na
                    soja; feriados e dias sem cotacao nao contam). --horizonte e --entrada
                    em dias uteis.

Na serie diaria as janelas dos modelos cobrem o mesmo tempo de calendario que na semanal
(1 semana = 5 dias uteis, 1 mes = 20, 3 meses = 60, 6 meses = 120, 1 ano = 240):
  PROF, PROFVAR  lags de 1, 2, 3, 4, 5, 10, 20, 60, 120 e 240 dias e media/desvio/min/max
                 moveis de 5, 10, 20, 60, 120 e 240 dias (na semanal: os do professor,
                 de 1 a 52 semanas). O diff_52 do script do professor continua sendo de 52
                 passos (52 dias uteis na diaria).
  RET            variacoes dos ultimos 5 dias e de 10 e 20 dias atras, variacao acumulada
                 em 5/20/60/120/240 dias, volatilidade em 5/20/60, distancia da media de
                 60/120/240 e sazonalidade pelo dia do ano.
  SNAIVE         preco do mesmo dia um ano antes (ou do ultimo dia com cotacao antes dele).
  CHRONOS        recebe a serie diaria, sem mudanca.

Previsao futura (modo passo; --sem-futuro desliga): depois do holdout, os modelos sao
retreinados com a serie inteira e preveem os --horizonte passos seguintes a ultima data
(previsao_futura.csv e 06_previsao_futura.png), ao lado do erro de cada um no holdout.

Para comparar as frequencias depois de varias rodadas: v5_diario_semanal/comparar.py.

Resultados em v5_diario_semanal/outputs[_soja][_janelaN][_entradaN]/<frequencia>/<modo>[_hNN].

Uso (a partir da pasta do projeto):
    .venv/bin/python v5_diario_semanal/previsao.py --produto soja --frequencia diaria --horizonte 5
    .venv/bin/python v5_diario_semanal/previsao.py --frequencia diaria --horizonte 5
    .venv/bin/python v5_diario_semanal/previsao.py --frequencia diaria --horizonte 20 --models RET,PROF
    .venv/bin/python v5_diario_semanal/previsao.py --frequencia semanal --horizonte 4   # = v4 com H=4
    .venv/bin/python v5_diario_semanal/previsao.py --frequencia diaria --modo cego
    .venv/bin/python v5_diario_semanal/previsao.py --frequencia diaria --entrada 20 --horizonte 5
    .venv/bin/python v5_diario_semanal/previsao.py --frequencia diaria --horizonte 5 --reavaliar
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "v3_backtest"))
sys.path.insert(0, str(RAIZ / "v4_holdout"))
import backtest as bt  # noqa: E402  (dados, modelos, metricas e estilo do v3)
import holdout as ho  # noqa: E402  (metricas, graficos e Chronos do holdout v4)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Janelas de cada frequencia. Na semanal, None = as do professor / as do v3 (iguais a v4).
FREQUENCIAS = {
    "semanal": {"arquivo": "semanal.csv", "unidade": "semana", "plural": "semanas", "por_ano": 52,
                "offset": "W-FRI", "prof_lags": None, "prof_janelas": None, "erro_movel": 13},
    "diaria": {"arquivo": "diario.csv", "unidade": "dia", "plural": "dias uteis", "por_ano": 240,
               "offset": "B",
               "prof_lags": [1, 2, 3, 4, 5, 10, 20, 60, 120, 240], "prof_janelas": [5, 10, 20, 60, 120, 240],
               "ret_lags": (0, 1, 2, 3, 4, 9, 19), "ret_acum": (5, 20, 60, 120, 240), "ret_vol": (5, 20, 60),
               "ret_media": (60, 120, 240), "erro_movel": 60},
}


def datas_futuras(datas, passos, offset):
    """As `passos` datas seguintes a ultima da serie: sextas (semanal) ou dias uteis (diaria;
    feriados nao sao conhecidos, entao entram como dia util)."""
    return pd.date_range(pd.DatetimeIndex(datas)[-1], periods=passos + 1, freq=offset)[1:]


# Onde ficam os CSVs de cada produto e qual coluna e o alvo.
PRODUTOS = {
    "milho": {"pastas": ("dados_milho",), "alvo": "media_pr", "gerar": "make dados"},
    "soja": {"pastas": ("dados_soja",), "alvo": "preco_rs", "gerar": "make dados-soja"},
}


def arquivo_dados(produto, cfg):
    p = PRODUTOS[produto]
    nome = f"{produto}_{cfg['arquivo']}"
    for pasta in p["pastas"]:
        if (RAIZ / pasta / nome).exists():
            return RAIZ / pasta / nome
    raise SystemExit(f"Nao achei {nome} em {', '.join(p['pastas'])}/. Gere com: {p['gerar']}")


def carregar_dados(arquivo, produto, target):
    """Milho: o carregador do v3 (pracas com poucas lacunas viram variaveis auxiliares).
    Soja: serie unica; as outras colunas (preco em US$) nao entram como variaveis."""
    if produto == "milho":
        return bt.carregar_dados(arquivo, target)
    df = pd.read_csv(arquivo, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    print(f"Linhas: {len(df)} | {df.data.min().date()} a {df.data.max().date()} | alvo: {target}")
    y_original = df[target].astype(float)
    return df.data, y_original.ffill(), y_original, pd.DataFrame(index=df.index)


# ============================================================
# FEATURES E MODELOS AJUSTADOS A FREQUENCIA
# ============================================================

def features_diretas(y, pracas, datas, cfg, entrada=None):
    """Features dos modelos RET_. Semanal: exatamente as do v3. Diaria: mesmas ideias, com as
    janelas em dias uteis (ver FREQUENCIAS) e sazonalidade pelo dia do ano. Sem pracas
    (soja), saem as duas variaveis de dispersao entre pracas."""
    f = bt.features_diretas(y, pracas, datas, entrada) if cfg["prof_lags"] is None else \
        features_diretas_diarias(y, pracas, datas, cfg, entrada)
    if pracas.shape[1] < 2:
        f = f.drop(columns=["dispersao_pracas", "amplitude_pracas"])
    return f


def features_diretas_diarias(y, pracas, datas, cfg, entrada=None):
    logy = np.log(y)
    r = logy.diff()
    f = pd.DataFrame(index=y.index)
    n_var = entrada - 1 if entrada else cfg["por_ano"]  # quantas variacoes diarias cabem na janela
    for k in (range(n_var) if entrada else cfg["ret_lags"]):
        f[f"ret_lag{k}"] = r.shift(k)
    for w in cfg["ret_acum"]:
        if w <= n_var:
            f[f"ret_acum_{w}"] = logy - logy.shift(w)
    for w in cfg["ret_vol"]:
        if w <= n_var:
            f[f"vol_{w}"] = r.rolling(w).std()
    for w in cfg["ret_media"]:
        if w <= n_var + 1:
            f[f"dist_media_{w}"] = logy - np.log(y.rolling(w).mean())
    dia = datas.dt.day_of_year.astype(float)
    f["sin_ano"] = np.sin(2 * np.pi * dia / 365.25)
    f["cos_ano"] = np.cos(2 * np.pi * dia / 365.25)
    f["dispersao_pracas"] = pracas.std(axis=1) / pracas.mean(axis=1)
    f["amplitude_pracas"] = np.log(pracas.max(axis=1) / pracas.min(axis=1))
    return f


def carregar_professor(cfg):
    """Script do professor com os lags e janelas moveis da frequencia (add_time_features le
    LAGS e ROLLING_WINDOWS do proprio modulo na hora de calcular)."""
    prof = bt.carregar_professor()
    if cfg["prof_lags"] is not None:
        prof.LAGS, prof.ROLLING_WINDOWS = list(cfg["prof_lags"]), list(cfg["prof_janelas"])
    return prof


def previsao_recursiva(modelo, prof, y, pracas, datas, origens, passos, cols, offset):
    """A mesma previsao recursiva do professor (prof.recursive_forecast), feita para todas as
    origens de uma vez: em cada passo calcula as features dele a partir do caminho
    (preco real ate a origem + previsoes anteriores) e preve o passo seguinte. Na serie
    diaria o original recalcularia as features da serie inteira a cada passo de cada origem
    (~1150 origens), o que levaria horas. Diferenca: a data de cada passo vem da propria
    serie (dias uteis com cotacao), e nao de um calendario fixo; so depois do fim da serie
    (previsao futura) as datas seguem o calendario `offset`. Retorna (origens, passos)."""
    y = y.to_numpy(float)
    aux = pracas.to_numpy(float)
    origens = np.asarray(origens)
    datas = pd.DatetimeIndex(datas)
    faltam = origens.max() + passos - (len(datas) - 1)
    if faltam > 0:
        datas = datas.append(datas_futuras(datas, faltam, offset))
    L = max(max(prof.LAGS), max(prof.ROLLING_WINDOWS), 53)  # 53: diff_52 do professor
    pos = origens[:, None] + np.arange(-L + 1, 1)
    caminho = np.full((len(origens), L + passos), np.nan)
    caminho[:, :L] = np.where(pos >= 0, y[np.clip(pos, 0, None)], np.nan)
    aux_o = aux[origens]  # aux_*_lag1 fica parado no ultimo valor conhecido, como no original
    for k in range(1, passos + 1):
        j = L - 1 + k  # coluna do passo k no caminho
        alvo = origens + k
        dia = datas[alvo]
        f = {"t": alvo.astype(float), "ano": dia.year.to_numpy(float),
             "sin_ano": np.sin(2 * np.pi * dia.day_of_year.to_numpy() / 365.25),
             "cos_ano": np.cos(2 * np.pi * dia.day_of_year.to_numpy() / 365.25),
             "diff_1": caminho[:, j - 1] - caminho[:, j - 2], "diff_52": caminho[:, j - 1] - caminho[:, j - 53]}
        for lag in prof.LAGS:
            f[f"lag_{lag}"] = caminho[:, j - lag]
        for w in prof.ROLLING_WINDOWS:
            jan = caminho[:, j - w: j]
            f[f"roll_mean_{w}"], f[f"roll_std_{w}"] = jan.mean(axis=1), jan.std(axis=1, ddof=1)
            f[f"roll_min_{w}"], f[f"roll_max_{w}"] = jan.min(axis=1), jan.max(axis=1)
        for i in range(aux.shape[1]):
            f[f"aux_{i + 1}_lag1"] = aux_o[:, i]
        caminho[:, j] = modelo.predict(np.column_stack([f[c] for c in cols]))
    return caminho[:, L:]


# ============================================================
# PREVISOES POR FAMILIA
# ============================================================

def prever_base(escolhidos, cfg, y, datas, origens, alvos, s, modo):
    if cfg["prof_lags"] is None:
        return ho.prever_base(escolhidos, y, origens, alvos, s, modo)
    linhas = [{"modelo": "NAIVE", "origem": o, "alvo": a, "pred": y.iloc[o]} for o, a in zip(origens, alvos)]
    if "SNAIVE" not in escolhidos:
        return linhas
    fim_treino = datas.iloc[s - 1]
    for o, a in zip(origens, alvos):
        # passo: mesmo dia do ano anterior; cego: volta quantos anos for preciso para cair no treino
        anos = 1
        while modo == "cego" and datas.iloc[a] - pd.DateOffset(years=anos) > fim_treino:
            anos += 1
        ref = int(np.searchsorted(datas, datas.iloc[a] - pd.DateOffset(years=anos), side="right")) - 1
        if ref >= 0:
            # com horizonte de ~1 ano a data de referencia pode passar da origem: usa o ultimo preco conhecido
            linhas.append({"modelo": "SNAIVE", "origem": o, "alvo": a, "pred": y.iloc[min(ref, o)]})
    return linhas


def prever_ret(nomes, cfg, y, y_original, pracas, datas, origens, alvos, s, h, ini=0, entrada=None):
    feats = features_diretas(y, pracas, datas, cfg, entrada)
    logy = np.log(y)
    alvo_ret = logy.shift(-h) - logy
    idx = np.arange(ini, s - h)  # da janela ate o ultimo rotulo (i+h) ainda dentro do treino
    treino = feats.iloc[idx].assign(_alvo=alvo_ret.iloc[idx])
    treino = treino[y_original.shift(-h).iloc[idx].notna().values].dropna()
    X, Y = treino.drop(columns="_alvo").to_numpy(), treino._alvo.to_numpy()
    x_teste = feats.iloc[origens]
    ok = x_teste.notna().all(axis=1).to_numpy()
    linhas = []
    for nome, modelo in bt.modelos_ret(nomes).items():
        t0 = time.perf_counter()
        modelo.fit(X, Y)
        ret = modelo.predict(x_teste[ok].to_numpy())
        base = y.iloc[np.asarray(origens)[ok]].to_numpy()
        for o, a, p in zip(np.asarray(origens)[ok], np.asarray(alvos)[ok], base * np.exp(ret)):
            linhas.append({"modelo": f"RET_{nome}", "origem": o, "alvo": a, "pred": p})
        print(f"  RET_{nome} ok [{time.perf_counter() - t0:.0f}s]")
    return linhas


def prever_professor(nomes, cfg, datas, y, y_original, pracas, origens, alvos, s, h, modo, variacao=False, ini=0,
                     entrada=None):
    prefixo = "PROFVAR" if variacao else "PROF"
    prof = carregar_professor(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        disponiveis = set(prof.build_models(nomes))
    faltando = [n for n in nomes if n not in disponiveis]
    if faltando:
        print(f"  Indisponiveis (instale xgboost/lightgbm): {faltando}")

    hist = pd.DataFrame({"timestamp": datas, "target": y})
    for i, c in enumerate(pracas.columns, start=1):
        hist[f"aux_{i}"] = pracas[c].values
    feats = prof.add_time_features(hist)
    feats.loc[y_original.isna(), "target"] = np.nan
    cols = bt.colunas_prof_entrada(prof.get_feature_columns(feats), entrada)
    treino = feats.iloc[ini:s].dropna(subset=cols + ["target"])

    linhas = []
    for nome in [n for n in nomes if n in disponiveis]:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            modelo = prof.build_models([nome])[nome]
        if variacao:
            modelo = bt.NivelViaVariacao(modelo, cols.index("lag_1"))
        modelo.fit(treino[cols].to_numpy(float), treino.target.to_numpy(float))
        if modo == "cego":
            preds = previsao_recursiva(modelo, prof, y, pracas, datas, [s - 1], len(alvos), cols, cfg["offset"])[0]
        elif h == 1:
            # as features da linha a ja usam so dados reais ate a-1: previsao de 1 passo
            preds = modelo.predict(feats.iloc[alvos][cols].ffill().to_numpy(float))
        else:
            preds = previsao_recursiva(modelo, prof, y, pracas, datas, origens, h, cols, cfg["offset"])[:, -1]
        for o, a, p in zip(origens, alvos, preds):
            linhas.append({"modelo": f"{prefixo}_{nome}", "origem": o, "alvo": a, "pred": float(p)})
        print(f"  {prefixo}_{nome} ok [{time.perf_counter() - t0:.0f}s]")
    return linhas


# ============================================================
# PREVISAO FUTURA (depois do fim da serie)
# ============================================================

def prever_futuro(args, cfg, escolhidos, chronos_ids, datas, y, y_original, pracas, h, t0):
    """Retreina os modelos com a serie INTEIRA (ou os ultimos --janela-anos) e preve os
    proximos h passos depois da ultima data. E o "uso de verdade" do modelo; o holdout so
    diz em quais modelos confiar. PROF/PROFVAR/CHRONOS dao o caminho passo a passo; RET
    (direto) so o ponto final h; as combinacoes saem onde todos os componentes existem."""
    n = len(y)
    ini = bt.inicio_janela(datas, n - 1, args.janela_anos)
    alvos = list(range(n, n + h))
    origens = [n - 1] * h
    escolhidos = [m for m in escolhidos if m != "SNAIVE"]

    def da_familia(f):
        return [m.split("_", 1)[1] for m in escolhidos if bt.familia(m) == f]

    linhas = [{"modelo": "NAIVE", "origem": n - 1, "alvo": a, "pred": y.iloc[-1]} for a in alvos]
    with contextlib.redirect_stdout(io.StringIO()):  # mesmas mensagens "ok" do holdout
        if da_familia("RET"):
            linhas += prever_ret(da_familia("RET"), cfg, y, y_original, pracas, datas, [n - 1], [n - 1 + h], n, h,
                                 ini, args.entrada)
        for fam, var in (("PROF", False), ("PROFVAR", True)):
            if da_familia(fam):
                linhas += prever_professor(da_familia(fam), cfg, datas, y, y_original, pracas, origens, alvos, n, h,
                                           "cego", variacao=var, ini=ini, entrada=args.entrada)
    for model_id in chronos_ids:
        nome = bt.nome_chronos(model_id)
        if nome in escolhidos:
            print(f"  {nome}... [{time.perf_counter() - t0:.0f}s]")
            linhas += ho.prever_chronos(model_id, nome, y, origens, alvos, n, h, "cego", args.entrada)

    fut = pd.DataFrame(linhas)
    for c in ("q10", "q90"):
        if c not in fut:
            fut[c] = np.nan
    fut["passo"] = fut.alvo - (n - 1)
    fut["data"] = datas_futuras(datas, h, cfg["offset"])[fut.passo.to_numpy() - 1]
    fut = bt.adicionar_combinacoes(fut.drop(columns=["origem", "alvo"]), ["passo"])
    fut["variacao_percent"] = 100 * (fut.pred / y.iloc[-1] - 1)
    return fut[["modelo", "passo", "data", "pred", "q10", "q90", "variacao_percent"]].sort_values(["modelo", "passo"])


def grafico_futuro(fut, metricas, datas, y_original, cfg, h, out):
    """Historico recente + previsao futura do NAIVE e do melhor modelo de cada familia no holdout."""
    destaque = ["NAIVE"] + [metricas[metricas.familia == f].iloc[0].modelo
                            for f in [f for f in bt.FAMILIAS if f != "BASE"]
                            if (metricas.familia == f).any() and metricas[metricas.familia == f].iloc[0].modelo
                            in set(fut.modelo)]
    rel = metricas.set_index("modelo").MAE_rel_naive
    inicio = datas.iloc[-1] - max(pd.Timedelta(days=90), 4 * (fut.data.max() - datas.iloc[-1]))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    mask = datas >= inicio
    ax.plot(datas[mask], y_original[mask], color=bt.COR_REAL, linewidth=1.6, label="Real")
    ultimo = (datas.iloc[-1], y_original.dropna().iloc[-1])
    for m in destaque:
        g = fut[fut.modelo == m].sort_values("passo")
        c = ho.cor_modelo(m)
        if g.q10.notna().any():
            ax.fill_between(g.data, g.q10, g.q90, color=c, alpha=0.15, linewidth=0)
        rotulo = m if m == "NAIVE" else f"{m} (erro rel. no holdout {rel.get(m, np.nan):.2f})"
        xs, ys = [ultimo[0]] + list(g.data), [ultimo[1]] + list(g.pred)
        ax.plot(xs, ys, color=c, linewidth=1.8, linestyle="--" if m == "NAIVE" or len(g) == 1 else "-",
                marker="o" if len(g) <= 30 else None, markersize=4, label=rotulo)
    ax.axvline(datas.iloc[-1], color="#c3c2b7", linewidth=1)
    plural = "dia(s) util(eis)" if cfg["unidade"] == "dia" else "semana(s)"
    bt.estilo(ax, f"Previsao para os proximos {h} {plural} apos {datas.iloc[-1].date()} "
                  "(modelos retreinados com a serie inteira; faixa = 10-90% do Chronos)", "R$/saca 60 kg")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "06_previsao_futura.png", dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--produto", choices=list(PRODUTOS), default="milho",
                        help="Serie usada para treinar e testar: milho (SIMA/DERAL) ou soja (CEPEA/ESALQ).")
    parser.add_argument("--frequencia", choices=list(FREQUENCIAS), default="semanal",
                        help="semanal (passos em semanas) ou diaria (passos em dias uteis com cotacao).")
    parser.add_argument("--data-file", default=None,
                        help="Padrao: dados_milho/milho_<frequencia>.csv ou dados_soja/soja_<frequencia>.csv")
    parser.add_argument("--target", default=None, help="Coluna alvo. Padrao: media_pr (milho) ou preco_rs (soja).")
    parser.add_argument("--treino", type=float, default=0.8, help="Fracao inicial da serie usada no treino.")
    parser.add_argument("--modo", choices=["passo", "cego"], default="passo")
    parser.add_argument("--horizonte", type=int, default=1,
                        help="Passos a frente (so no modo passo): semanas (1 a 52) ou dias uteis (1 a 240).")
    parser.add_argument("--models", default="ALL",
                        help="ALL, familias (BASE, PROF, PROFVAR, RET, CHRONOS) e/ou nomes. NAIVE sempre roda.")
    parser.add_argument("--chronos-models", default="amazon/chronos-bolt-small,amazon/chronos-2")
    parser.add_argument("--output-dir", default=None,
                        help="Padrao: v5_diario_semanal/outputs[_soja][_janelaN][_entradaN]/<frequencia>/<modo>[_hNN]")
    parser.add_argument("--entrada", type=int, default=None,
                        help="Janela de entrada: cada previsao so ve os precos dos ultimos N passos.")
    parser.add_argument("--janela-anos", type=int, default=None,
                        help="Janela deslizante: treina so com os ultimos N anos antes do corte.")
    parser.add_argument("--reavaliar", action="store_true",
                        help="Nao roda modelos: recalcula metricas, combinacoes e graficos do CSV salvo.")
    parser.add_argument("--limpar", action="store_true",
                        help="Descarta os resultados anteriores (por padrao, sao mantidos os modelos que nao rodaram).")
    parser.add_argument("--sem-futuro", action="store_true",
                        help="Nao faz a previsao futura (modelos retreinados com tudo, h passos apos a ultima data).")
    args = parser.parse_args()

    cfg = FREQUENCIAS[args.frequencia]
    unid, plural = cfg["unidade"], cfg["plural"]
    if not 0.5 <= args.treino < 1:
        parser.error("--treino deve estar entre 0.5 e 0.99")
    if args.modo == "passo" and not 1 <= args.horizonte <= cfg["por_ano"]:
        parser.error(f"--horizonte deve estar entre 1 e {cfg['por_ano']} ({plural})")
    if args.entrada is not None and args.entrada < 1:
        parser.error("--entrada deve ser pelo menos 1")
    h = 0 if args.modo == "cego" else args.horizonte
    pasta = "cego" if args.modo == "cego" else f"passo_h{h:02d}"
    produto = "" if args.produto == "milho" else f"_{args.produto}"  # milho fica onde ja estava
    variante = f"outputs{produto}{bt.sufixo_saida(args.janela_anos, args.entrada)}"
    out = Path(args.output_dir) if args.output_dir else AQUI / variante / args.frequencia / pasta
    out.mkdir(parents=True, exist_ok=True)

    chronos_ids = [m.strip() for m in args.chronos_models.split(",") if m.strip()]
    todos = bt.catalogo(chronos_ids)
    escolhidos = bt.selecionar_modelos(args.models, todos)
    if args.modo == "cego" and any(bt.familia(m) == "RET" for m in escolhidos):
        print("Modo cego: modelos RET_ ficam de fora (preveem um horizonte fixo).")
        escolhidos = [m for m in escolhidos if bt.familia(m) != "RET"]

    datas, y, y_original, pracas = carregar_dados(args.data_file or arquivo_dados(args.produto, cfg), args.produto,
                                                  args.target or PRODUTOS[args.produto]["alvo"])
    n = len(y)
    s = int(round(n * args.treino))
    ini = bt.inicio_janela(datas, s - 1, args.janela_anos)
    print(f"Produto: {args.produto} | Frequencia: {args.frequencia} "
          f"(1 passo = 1 {unid}{' util com cotacao' if unid == 'dia' else ''})")
    print(f"Treino: {datas.iloc[ini].date()} a {datas.iloc[s - 1].date()} ({s - ini} {plural})"
          + (f" | janela deslizante de {args.janela_anos} anos" if args.janela_anos else ""))
    print("Janela de entrada: " + (f"precos dos ultimos {args.entrada} {plural}" if args.entrada
                                   else "padrao de cada modelo (ate 1 ano)"))
    print(f"Teste:  {datas.iloc[s].date()} a {datas.iloc[-1].date()} ({n - s} {plural}) | modo {args.modo}"
          + (f", {h} {plural} a frente" if h else ""))
    print(f"Modelos: {', '.join(escolhidos)}")

    alvos = list(range(s, n))
    origens = [s - 1] * len(alvos) if args.modo == "cego" else [a - h for a in alvos]

    t0 = time.perf_counter()
    arquivo = out / "previsoes.csv"
    if args.reavaliar:
        prev = bt.ler_previsoes(arquivo)
        print(f"Reavaliando {arquivo} ({prev.modelo.nunique()} modelos), sem rodar nada.")
    else:
        prev = rodar_modelos(args, cfg, escolhidos, chronos_ids, datas, y, y_original, pracas, origens, alvos, s, h,
                             t0, ini)
        if not args.limpar:
            prev = bt.mesclar_com_anterior(prev, arquivo, ["origem", "alvo"])
    prev = bt.adicionar_combinacoes(prev, ["origem", "alvo"])

    metricas = ho.calcular_metricas(prev, h)
    prev.to_csv(arquivo, index=False)
    metricas.to_csv(out / "metricas.csv", index=False, float_format="%.4f")

    prev_ok = prev.dropna(subset=["real"])
    ho.grafico_visao_geral(prev_ok, metricas, datas, y_original, s, args.modo, h, out, ini, unid)
    ho.grafico_todos(prev_ok, metricas, datas, y_original, s, args.modo, h, out, unid)
    ho.grafico_ranking(metricas, out)
    ho.grafico_erro_no_tempo(prev_ok, metricas, datas, out, cfg["erro_movel"], plural)
    bt.grafico_prof_nivel_vs_variacao(metricas, out, "05_professor_preco_vs_variacao.png")

    cols = ["modelo", "MAE", "MAPE_percent", "MAE_rel_naive", "acerto_direcao_percent", "dm_pvalor", "vs_naive"]
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 180):
        print("\nResultado no periodo de teste (ordenado pelo erro relativo ao ingenuo):")
        print(metricas[cols].to_string(index=False))

    arquivo_fut = out / "previsao_futura.csv"
    fut = None
    if args.modo == "passo" and not args.reavaliar and not args.sem_futuro:
        print(f"\nPrevisao futura: {h} {plural} apos {datas.iloc[-1].date()}, modelos retreinados com a serie inteira...")
        fut = prever_futuro(args, cfg, escolhidos, chronos_ids, datas, y, y_original, pracas, h, t0)
        if arquivo_fut.exists() and not args.limpar:
            ant = pd.read_csv(arquivo_fut, parse_dates=["data"])
            if set(ant.data) == set(fut.data):  # mesma data final e horizonte: mantem os que nao rodaram
                manter = ant[~ant.modelo.isin(set(fut.modelo)) & (ant.modelo.map(bt.familia) != "COMBO")]
                fut = bt.adicionar_combinacoes(pd.concat([fut, manter], ignore_index=True), ["passo"])
                fut["variacao_percent"] = 100 * (fut.pred / y.iloc[-1] - 1)  # as combinacoes herdam a do NAIVE
        fut.to_csv(arquivo_fut, index=False, float_format="%.4f")
    elif args.modo == "passo" and arquivo_fut.exists():
        fut = pd.read_csv(arquivo_fut, parse_dates=["data"])
    if fut is not None:
        grafico_futuro(fut, metricas, datas, y_original, cfg, h, out)
        final = (fut[fut.passo == h].merge(metricas[["modelo", "MAE_rel_naive", "vs_naive"]], on="modelo", how="left")
                 .sort_values("MAE_rel_naive"))
        with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 180):
            print(f"\nPrevisao para {final.data.iloc[0].date()} (ultimo preco: R$ {y.iloc[-1]:.2f}), "
                  "ordenada pelo erro no holdout:")
            print(final[["modelo", "pred", "variacao_percent", "q10", "q90", "MAE_rel_naive", "vs_naive"]]
                  .to_string(index=False))
    print(f"\nTempo total: {time.perf_counter() - t0:.0f}s | Saidas: {out.resolve()}")


def rodar_modelos(args, cfg, escolhidos, chronos_ids, datas, y, y_original, pracas, origens, alvos, s, h, t0, ini=0):
    def da_familia(f):
        return [m.split("_", 1)[1] for m in escolhidos if bt.familia(m) == f]

    linhas = prever_base(escolhidos, cfg, y, datas, origens, alvos, s, args.modo)
    if da_familia("RET"):
        print("Modelos de retorno (diretos)...")
        linhas += prever_ret(da_familia("RET"), cfg, y, y_original, pracas, datas, origens, alvos, s, h, ini,
                             args.entrada)
    if da_familia("PROF"):
        print(f"Modelos do professor... [{time.perf_counter() - t0:.0f}s]")
        linhas += prever_professor(da_familia("PROF"), cfg, datas, y, y_original, pracas, origens, alvos, s, h,
                                   args.modo, ini=ini, entrada=args.entrada)
    if da_familia("PROFVAR"):
        print(f"Modelos do professor prevendo variacao... [{time.perf_counter() - t0:.0f}s]")
        linhas += prever_professor(da_familia("PROFVAR"), cfg, datas, y, y_original, pracas, origens, alvos, s, h,
                                   args.modo, variacao=True, ini=ini, entrada=args.entrada)
    for model_id in chronos_ids:
        nome = bt.nome_chronos(model_id)
        if nome in escolhidos:
            print(f"{nome}... [{time.perf_counter() - t0:.0f}s]")
            linhas += ho.prever_chronos(model_id, nome, y, origens, alvos, s, h, args.modo, args.entrada)

    prev = pd.DataFrame(linhas)
    for c in ("q10", "q90"):
        if c not in prev:
            prev[c] = np.nan
    prev["h"] = h
    prev["data_origem"] = datas.iloc[prev.origem].values
    prev["data_alvo"] = datas.iloc[prev.alvo].values
    prev["y_origem"] = y.iloc[prev.origem].values
    prev["real"] = y_original.iloc[prev.alvo].values
    return prev


if __name__ == "__main__":
    main()

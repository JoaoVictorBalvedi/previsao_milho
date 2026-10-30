"""Teste holdout: treina com os primeiros 80% da serie e preve os 20% finais,
que ja aconteceram, para comparar com o preco real.

Com a base atual: treino de 2001 ate jan/2021, teste de fev/2021 ate dez/2025.

Dois modos:
  passo (padrao)  Os modelos sao treinados UMA vez, so com os 80%. Depois, em cada
                  semana do teste, preveem o preco --horizonte semanas a frente usando
                  o historico real ate aquela semana (como no uso real: toda semana
                  chega um preco novo, mas o modelo nao e retreinado).
  cego            Uma unica previsao feita no fim dos 80% para todo o periodo de teste
                  (~5 anos), sem nunca ver os precos reais do teste. Os modelos RET_
                  ficam de fora (eles preveem um horizonte fixo).

Os modelos sao os mesmos do backtest (v3_backtest/backtest.py): NAIVE, SNAIVE,
PROF_* (professor), PROFVAR_* (professor prevendo a variacao), RET_* (retorno),
CHRONOS_* e as combinacoes COMBO_*. O Chronos nao e treinado: em cada previsao ele
recebe so o historico ate aquela data.

Janela de treino: por padrao o treino vai do inicio da serie (2001) ate o corte dos 80%.
Com --janela-anos N ele usa so os ultimos N anos antes do corte (janela deslizante) e os
resultados vao para outputs_janelaN/. O Chronos nao e treinado e continua vendo tudo.

Janela de entrada: --entrada N limita o que o modelo ENXERGA em cada previsao aos precos
das ultimas N semanas (vale tambem para o Chronos). Resultados em outputs_entradaN/.

Por padrao, modelos que nao rodaram agora sao mantidos da rodada anterior (mesmo
modo/horizonte/treino), entao da para rodar so um modelo novo. --limpar comeca do zero
e --reavaliar so recalcula metricas, combinacoes e graficos a partir do CSV salvo.

Uso (a partir da pasta do projeto):
    .venv/bin/python v4_holdout/holdout.py                    # passo, 1 semana a frente
    .venv/bin/python v4_holdout/holdout.py --horizonte 4      # 4 semanas a frente
    .venv/bin/python v4_holdout/holdout.py --modo cego
    .venv/bin/python v4_holdout/holdout.py --models PROF --treino 0.7
    .venv/bin/python v4_holdout/holdout.py --models PROFVAR          # acrescenta so os PROFVAR
    .venv/bin/python v4_holdout/holdout.py --reavaliar               # sem rodar modelos
    .venv/bin/python v4_holdout/holdout.py --janela-anos 10          # treina so com 2011-2021
    .venv/bin/python v4_holdout/holdout.py --entrada 4 --horizonte 2 # olha 4 semanas, preve 2
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
import backtest as bt  # noqa: E402  (reaproveita dados, modelos, metricas e estilo do v3)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================
# PREVISOES POR FAMILIA
# ============================================================

def prever_base(escolhidos, y, origens, alvos, s, modo):
    linhas = []
    for o, a in zip(origens, alvos):
        linhas.append({"modelo": "NAIVE", "origem": o, "alvo": a, "pred": y.iloc[o]})
        if "SNAIVE" in escolhidos:
            # cego: repete o ultimo ano do treino; passo: mesma semana do ano anterior
            ref = s - 52 + (a - s) % 52 if modo == "cego" else a - 52
            linhas.append({"modelo": "SNAIVE", "origem": o, "alvo": a, "pred": y.iloc[ref]})
    return linhas


def prever_ret(nomes, y, y_original, pracas, datas, origens, alvos, s, h, ini=0, entrada=None):
    feats = bt.features_diretas(y, pracas, datas, entrada)
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


def prever_professor(nomes, datas, y, y_original, pracas, origens, alvos, s, h, modo, variacao=False, ini=0,
                     entrada=None):
    prefixo = "PROFVAR" if variacao else "PROF"
    prof = bt.carregar_professor()
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
            fut = prof.recursive_forecast(modelo, hist.iloc[:s], cols, len(alvos), "W-FRI")
            preds = fut.previsao_target.to_numpy()
        elif h == 1:
            # as features da linha a ja usam so dados reais ate a-1: previsao de 1 passo
            preds = modelo.predict(feats.iloc[alvos][cols].ffill().to_numpy(float))
        else:
            preds = [prof.recursive_forecast(modelo, hist.iloc[: o + 1], cols, h, "W-FRI").previsao_target.iloc[-1]
                     for o in origens]
        for o, a, p in zip(origens, alvos, preds):
            linhas.append({"modelo": f"{prefixo}_{nome}", "origem": o, "alvo": a, "pred": float(p)})
        print(f"  {prefixo}_{nome} ok [{time.perf_counter() - t0:.0f}s]")
    return linhas


def prever_chronos(model_id, nome, y, origens, alvos, s, h, modo, entrada=None):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    if modo == "cego":
        contextos, passos = [bt.contexto(y, s - 1, entrada)], len(alvos)
    else:
        contextos, passos = [bt.contexto(y, o, entrada) for o in origens], h
    # processo separado: a libomp do torch conflita com a do XGBoost/LightGBM (ver v3)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        q, _ = ex.submit(bt.rodar_chronos, model_id, contextos, passos, y.iloc[:s].to_numpy(), 1).result()
    linhas = []
    for i, (o, a) in enumerate(zip(origens, alvos)):
        qi = q[0, i] if modo == "cego" else q[i, h - 1]
        linhas.append({"modelo": nome, "origem": o, "alvo": a, "pred": qi[1], "q10": qi[0], "q90": qi[2]})
    return linhas


# ============================================================
# METRICAS
# ============================================================

def calcular_metricas(prev, h):
    """Uma linha por modelo no periodo de teste (o NAIVE e comparado na mesma data alvo).
    h = 0 no modo cego."""
    naive = prev[prev.modelo == "NAIVE"].set_index("alvo").pred
    linhas = []
    for modelo, g in prev.dropna(subset=["real"]).sort_values("alvo").groupby("modelo"):
        erro = g.real - g.pred
        mae = erro.abs().mean()
        mae_naive = (g.real - naive.loc[g.alvo].values).abs().mean()
        var_real, var_prev = g.real - g.y_origem, g.pred - g.y_origem
        mov = var_real != 0
        mse = (erro ** 2).mean()
        mse_naive = ((g.real - naive.loc[g.alvo].values) ** 2).mean()
        linha = {"modelo": modelo, "familia": bt.familia(modelo), "n": len(g), "MAE": mae,
                 "MSE": mse, "RMSE": np.sqrt(mse),
                 "MAPE_percent": 100 * (erro.abs() / g.real).mean(),
                 "MAE_rel_naive": mae / mae_naive,
                 "RMSE_rel_naive": np.sqrt(mse / mse_naive),
                 "acerto_direcao_percent": np.nan if modelo == "NAIVE" else
                 100 * np.mean(np.sign(var_real[mov]) == np.sign(var_prev[mov]))}
        d = erro.abs().values - (g.real - naive.loc[g.alvo].values).abs().values
        # passo: previsoes de h semanas se sobrepoem em h-1 vizinhas; cego: regra de Newey-West
        lag = h - 1 if h else int(4 * (len(g) / 100) ** (2 / 9))
        linha["dm_pvalor"] = bt.diebold_mariano(d, lag)
        linha["vs_naive"] = bt.comparacao_naive(linha["MAE_rel_naive"], linha["dm_pvalor"])
        if g.q10.notna().any():
            linha["cobertura_80_percent"] = 100 * np.mean((g.real >= g.q10) & (g.real <= g.q90))
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("MAE_rel_naive").reset_index(drop=True)


# ============================================================
# GRAFICOS
# ============================================================

def cor_modelo(m):
    return bt.COR_NAIVE if m == "NAIVE" else bt.COR_FAMILIA[bt.familia(m)]


def titulo_teste(modo, h, unidade="semana"):
    """unidade: "semana" ou "dia" (dia util com cotacao, usado na v5)."""
    plural = "dia(s) util(eis)" if unidade == "dia" else "semana(s)"
    return "previsao unica feita no fim do treino" if modo == "cego" else \
        f"previsao {h} {plural} a frente, com o historico real ate cada {unidade}"


def grafico_visao_geral(prev, metricas, datas, y_original, s, modo, h, out, ini=0, unidade="semana"):
    destaque = ["NAIVE"] + [metricas[metricas.familia == f].iloc[0].modelo
                            for f in [f for f in bt.FAMILIAS if f != "BASE"] if (metricas.familia == f).any()]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axvspan(datas.iloc[s], datas.iloc[-1], color="#e6e5e0", alpha=0.6, linewidth=0)
    ax.plot(datas, y_original, color=bt.COR_REAL, linewidth=1.6, label="Real")
    rel = metricas.set_index("modelo").MAE_rel_naive
    for m in destaque:
        g = prev[prev.modelo == m].sort_values("alvo")
        ax.plot(datas.iloc[g.alvo], g.pred, color=cor_modelo(m), linewidth=1.6,
                linestyle="--" if m == "NAIVE" else "-",
                label=m if m == "NAIVE" else f"{m} (erro rel. {rel[m]:.2f})")
    if ini:
        ax.axvspan(datas.iloc[0], datas.iloc[ini], color="#f4f3ef", linewidth=0)
        ax.annotate("fora da janela de treino", (datas.iloc[0], 0.97), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points", fontsize=9, color="#52514e", va="top")
    bt.estilo(ax, f"Treino ({datas.iloc[ini].year}-{datas.iloc[s - 1].year}) e teste "
                  f"({datas.iloc[s].year}-{datas.iloc[-1].year}): {titulo_teste(modo, h, unidade)}", "R$/saca 60 kg")
    ax.annotate("teste (20% finais)", (datas.iloc[s], 0.97), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color="#52514e", va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "01_visao_geral.png", dpi=160)
    plt.close(fig)


def grafico_todos(prev, metricas, datas, y_original, s, modo, h, out, unidade="semana"):
    modelos = [m for m in metricas.sort_values("MAE_rel_naive").modelo]
    ncol = 4
    nlin = int(np.ceil(len(modelos) / ncol))
    fig, axes = plt.subplots(nlin, ncol, figsize=(4.2 * ncol, 2.9 * nlin), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    teste = datas.iloc[s:]
    met = metricas.set_index("modelo")
    for ax, m in zip(axes, modelos):
        g = prev[prev.modelo == m].sort_values("alvo")
        c = cor_modelo(m)
        if g.q10.notna().any():
            ax.fill_between(datas.iloc[g.alvo], g.q10, g.q90, color=c, alpha=0.18, linewidth=0)
        ax.plot(teste, y_original.iloc[s:], color=bt.COR_REAL, linewidth=1.2)
        ax.plot(datas.iloc[g.alvo], g.pred, color=c, linewidth=1.3, linestyle="--" if m == "NAIVE" else "-")
        bt.estilo(ax, f"{m}\nMAPE {met.MAPE_percent[m]:.1f}% | erro rel. {met.MAE_rel_naive[m]:.2f}", "")
        ax.title.set_fontsize(9)
        ax.tick_params(labelsize=8)
    for ax in axes[len(modelos):]:
        ax.set_visible(False)
    fig.suptitle(f"Todos os modelos no periodo de teste (preto = real), {titulo_teste(modo, h, unidade)}. "
                 "Ordem: do menor para o maior erro.", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "02_todos_os_modelos.png", dpi=140)
    plt.close(fig)


def grafico_ranking(metricas, out):
    g = metricas[~metricas.modelo.isin(["NAIVE"])].sort_values("MAE_rel_naive")
    fora = g[g.MAE_rel_naive > 3]
    g = g[g.MAE_rel_naive <= 3]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(g) + 2.4))
    ax.barh(g.modelo, g.MAE_rel_naive, color=[cor_modelo(m) for m in g.modelo], height=0.7,
            edgecolor="white", linewidth=2)
    ax.axvline(1, color=bt.COR_REAL, linewidth=1)
    for i, (v, p) in enumerate(zip(g.MAE_rel_naive, g.dm_pvalor)):
        ax.text(v, i, f" {v:.2f}" + ("*" if p < 0.05 else ""), va="center", fontsize=8, color="#52514e")
    ax.invert_yaxis()
    ax.set_xlim(0, g.MAE_rel_naive.max() * 1.15)
    import textwrap
    nota = ("" if fora.empty else "\n" + textwrap.fill(
        "Fora do grafico (erro muito alto): "
        + ", ".join(f"{m} {v:.1f}" for m, v in zip(fora.modelo, fora.MAE_rel_naive)), 110))
    bt.estilo(ax, "Erro relativo ao ingenuo (< 1 = melhor; * = diferenca significativa, p < 0,05)" + nota, "")
    ax.title.set_fontsize(10)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    bt.legenda_familias(fig, [f for f in [f for f in bt.FAMILIAS if f != "BASE"] if (metricas.familia == f).any()])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "03_erro_relativo.png", dpi=160)
    plt.close(fig)


def grafico_erro_no_tempo(prev, metricas, datas, out, janela=13, unidade="semanas"):
    """Erro medio movel (padrao: 13 semanas): mostra em que periodos cada modelo falha."""
    destaque = ["NAIVE"] + [metricas[metricas.familia == f].iloc[0].modelo
                            for f in [f for f in bt.FAMILIAS if f != "BASE"] if (metricas.familia == f).any()]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for m in destaque:
        g = prev[prev.modelo == m].sort_values("alvo")
        erro = (100 * (g.real - g.pred).abs() / g.real).rolling(janela, min_periods=max(2, janela // 3)).mean()
        ax.plot(datas.iloc[g.alvo], erro, color=cor_modelo(m), linewidth=1.8,
                linestyle="--" if m == "NAIVE" else "-", label=m)
    bt.estilo(ax, f"Erro percentual medio das ultimas {janela} {unidade} ao longo do teste", "Erro (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "04_erro_ao_longo_do_tempo.png", dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", default=RAIZ / "dados_milho" / "milho_semanal.csv")
    parser.add_argument("--target", default="media_pr")
    parser.add_argument("--treino", type=float, default=0.8, help="Fracao inicial da serie usada no treino.")
    parser.add_argument("--modo", choices=["passo", "cego"], default="passo")
    parser.add_argument("--horizonte", type=int, default=1, help="Semanas a frente (so no modo passo).")
    parser.add_argument("--models", default="ALL",
                        help="ALL, familias (BASE, PROF, PROFVAR, RET, CHRONOS) e/ou nomes. NAIVE sempre roda.")
    parser.add_argument("--chronos-models", default="amazon/chronos-bolt-small,amazon/chronos-2")
    parser.add_argument("--output-dir", default=None, help="Padrao: v4_holdout/outputs[_janelaN][_entradaN]/<modo>[_hNN]")
    parser.add_argument("--entrada", type=int, default=None,
                        help="Janela de entrada: cada previsao so ve os precos das ultimas N semanas.")
    parser.add_argument("--janela-anos", type=int, default=None,
                        help="Janela deslizante: treina so com os ultimos N anos antes do corte.")
    parser.add_argument("--reavaliar", action="store_true",
                        help="Nao roda modelos: recalcula metricas, combinacoes e graficos do CSV salvo.")
    parser.add_argument("--limpar", action="store_true",
                        help="Descarta os resultados anteriores (por padrao, sao mantidos os modelos que nao rodaram).")
    args = parser.parse_args()

    if not 0.5 <= args.treino < 1:
        parser.error("--treino deve estar entre 0.5 e 0.99")
    if args.modo == "passo" and not 1 <= args.horizonte <= 52:
        parser.error("--horizonte deve estar entre 1 e 52")
    h = 0 if args.modo == "cego" else args.horizonte
    pasta = "cego" if args.modo == "cego" else f"passo_h{h:02d}"
    if args.entrada is not None and args.entrada < 1:
        parser.error("--entrada deve ser pelo menos 1")
    variante = f"outputs{bt.sufixo_saida(args.janela_anos, args.entrada)}"
    out = Path(args.output_dir) if args.output_dir else AQUI / variante / pasta
    out.mkdir(parents=True, exist_ok=True)

    chronos_ids = [m.strip() for m in args.chronos_models.split(",") if m.strip()]
    todos = bt.catalogo(chronos_ids)
    escolhidos = bt.selecionar_modelos(args.models, todos)
    if args.modo == "cego" and any(bt.familia(m) == "RET" for m in escolhidos):
        print("Modo cego: modelos RET_ ficam de fora (preveem um horizonte fixo).")
        escolhidos = [m for m in escolhidos if bt.familia(m) != "RET"]

    datas, y, y_original, pracas = bt.carregar_dados(args.data_file, args.target)
    n = len(y)
    s = int(round(n * args.treino))
    ini = bt.inicio_janela(datas, s - 1, args.janela_anos)
    print(f"Treino: {datas.iloc[ini].date()} a {datas.iloc[s - 1].date()} ({s - ini} semanas)"
          + (f" | janela deslizante de {args.janela_anos} anos" if args.janela_anos else ""))
    print("Janela de entrada: " + (f"precos das ultimas {args.entrada} semanas" if args.entrada
                                   else "padrao de cada modelo (ate 52 semanas)"))
    print(f"Teste:  {datas.iloc[s].date()} a {datas.iloc[-1].date()} ({n - s} semanas) | modo {args.modo}"
          + (f", {h} semana(s) a frente" if h else ""))
    print(f"Modelos: {', '.join(escolhidos)}")

    alvos = list(range(s, n))
    origens = [s - 1] * len(alvos) if args.modo == "cego" else [a - h for a in alvos]

    t0 = time.perf_counter()
    arquivo = out / "previsoes.csv"
    if args.reavaliar:
        prev = bt.ler_previsoes(arquivo)
        print(f"Reavaliando {arquivo} ({prev.modelo.nunique()} modelos), sem rodar nada.")
    else:
        prev = rodar_modelos(args, escolhidos, chronos_ids, datas, y, y_original, pracas, origens, alvos, s, h, t0,
                             ini)
        if not args.limpar:
            prev = bt.mesclar_com_anterior(prev, arquivo, ["origem", "alvo"])
    prev = bt.adicionar_combinacoes(prev, ["origem", "alvo"])

    metricas = calcular_metricas(prev, h)
    prev.to_csv(arquivo, index=False)
    metricas.to_csv(out / "metricas.csv", index=False, float_format="%.4f")

    prev_ok = prev.dropna(subset=["real"])
    grafico_visao_geral(prev_ok, metricas, datas, y_original, s, args.modo, h, out, ini)
    grafico_todos(prev_ok, metricas, datas, y_original, s, args.modo, h, out)
    grafico_ranking(metricas, out)
    grafico_erro_no_tempo(prev_ok, metricas, datas, out)
    bt.grafico_prof_nivel_vs_variacao(metricas, out, "05_professor_preco_vs_variacao.png")

    cols = ["modelo", "MAE", "MAPE_percent", "MAE_rel_naive", "acerto_direcao_percent", "dm_pvalor", "vs_naive"]
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 180):
        print("\nResultado no periodo de teste (ordenado pelo erro relativo ao ingenuo):")
        print(metricas[cols].to_string(index=False))
    print(f"\nTempo total: {time.perf_counter() - t0:.0f}s | Saidas: {out.resolve()}")


def rodar_modelos(args, escolhidos, chronos_ids, datas, y, y_original, pracas, origens, alvos, s, h, t0, ini=0):
    def da_familia(f):
        return [m.split("_", 1)[1] for m in escolhidos if bt.familia(m) == f]

    linhas = prever_base(escolhidos, y, origens, alvos, s, args.modo)
    if da_familia("RET"):
        print("Modelos de retorno (diretos)...")
        linhas += prever_ret(da_familia("RET"), y, y_original, pracas, datas, origens, alvos, s, h, ini,
                             args.entrada)
    if da_familia("PROF"):
        print("Modelos do professor...")
        linhas += prever_professor(da_familia("PROF"), datas, y, y_original, pracas, origens, alvos, s, h, args.modo,
                                   ini=ini, entrada=args.entrada)
    if da_familia("PROFVAR"):
        print("Modelos do professor prevendo variacao...")
        linhas += prever_professor(da_familia("PROFVAR"), datas, y, y_original, pracas, origens, alvos, s, h,
                                   args.modo, variacao=True, ini=ini, entrada=args.entrada)
    for model_id in chronos_ids:
        nome = bt.nome_chronos(model_id)
        if nome in escolhidos:
            print(f"{nome}... [{time.perf_counter() - t0:.0f}s]")
            linhas += prever_chronos(model_id, nome, y, origens, alvos, s, h, args.modo, args.entrada)

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

"""Compara as rodadas da v5 entre as frequencias diaria e semanal, por horizonte.

Le todos os <pasta>/<frequencia>/passo_hNN/metricas.csv que ja existem (rode antes, por
exemplo, make v5 FREQ=diaria H=1,5,20,60 e make v5 FREQ=semanal H=1,4,12) e converte o
horizonte para semanas de calendario (5 dias uteis = 1 semana), para responder:
  - o erro cresce do mesmo jeito com o horizonte nas duas series?
  - prever com dados diarios ganha do ingenuo onde o semanal nao ganha (ou o contrario)?

Atencao: os alvos nao sao identicos. Na semanal o alvo e a MEDIA da semana (mais suave);
na diaria e o preco do dia. Por isso o MAPE da semanal tende a ser um pouco menor no mesmo
horizonte; o erro relativo ao ingenuo (cada um contra o seu NAIVE) e a comparacao justa.

Saidas em <pasta>: comparacao_frequencias.csv e comparacao_frequencias.png.

Uso:
    .venv/bin/python v5_diario_semanal/comparar.py                       # pasta outputs/
    .venv/bin/python v5_diario_semanal/comparar.py --pasta v5_diario_semanal/outputs_janela10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "v3_backtest"))
import backtest as bt  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIAS_POR_SEMANA = 5
COR_FREQ = {"diaria": "#eb6834", "semanal": "#2a78d6"}


def ler_rodadas(pasta):
    tabelas = []
    for arq in sorted(pasta.glob("*/passo_h*/metricas.csv")):
        freq, h = arq.parent.parent.name, int(arq.parent.name.split("_h")[1])
        m = pd.read_csv(arq).assign(frequencia=freq, h=h)
        m["h_semanas"] = h / DIAS_POR_SEMANA if freq == "diaria" else float(h)
        tabelas.append(m)
    if not tabelas:
        raise SystemExit(f"Nenhum metricas.csv em {pasta}/<frequencia>/passo_h*/. Rode make v5 antes.")
    return pd.concat(tabelas, ignore_index=True)


def melhores(todas):
    """NAIVE e o melhor modelo (menor erro relativo) de cada frequencia e horizonte."""
    linhas = []
    for (freq, h), g in todas.groupby(["frequencia", "h"]):
        naive = g[g.modelo == "NAIVE"].iloc[0]
        top = g[g.modelo != "NAIVE"].sort_values("MAE_rel_naive").iloc[0]
        linhas.append({"frequencia": freq, "h": h, "h_semanas": naive.h_semanas,
                       "MAPE_naive": naive.MAPE_percent, "melhor_modelo": top.modelo,
                       "MAPE_melhor": top.MAPE_percent, "MAE_rel_naive_melhor": top.MAE_rel_naive,
                       "vs_naive_melhor": top.vs_naive, "n_modelos": len(g)})
    return pd.DataFrame(linhas).sort_values(["h_semanas", "frequencia"]).reset_index(drop=True)


def grafico(resumo, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for freq, g in resumo.groupby("frequencia"):
        c = COR_FREQ.get(freq, "#52514e")
        ax1.plot(g.h_semanas, g.MAPE_naive, "--", color=c, linewidth=1.8, marker="o", markersize=4,
                 label=f"{freq}: NAIVE")
        ax1.plot(g.h_semanas, g.MAPE_melhor, "-", color=c, linewidth=2, marker="o", markersize=5,
                 label=f"{freq}: melhor modelo")
        ax2.plot(g.h_semanas, g.MAE_rel_naive_melhor, "-", color=c, linewidth=2, marker="o", markersize=6,
                 label=freq)
        for x, v, nome, sig in zip(g.h_semanas, g.MAE_rel_naive_melhor, g.melhor_modelo, g.vs_naive_melhor):
            ax2.annotate(nome + ("*" if str(sig).startswith("melhor") else ""), (x, v), xytext=(4, 4),
                         textcoords="offset points", fontsize=7, color="#52514e", rotation=20)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ticks = sorted(set(resumo.h_semanas))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.legend(frameon=False, fontsize=9)
    ax2.axhline(1, color=bt.COR_REAL, linewidth=1)
    bt.estilo(ax1, "Erro medio (MAPE) por horizonte", "MAPE (%)", "Horizonte (semanas; 5 dias uteis = 1)")
    bt.estilo(ax2, "Melhor modelo: erro relativo ao ingenuo (< 1 = melhor; * = p < 0,05)", "MAE / MAE do NAIVE",
              "Horizonte (semanas; 5 dias uteis = 1)")
    fig.suptitle("Holdout 80/20: serie diaria x semanal", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "comparacao_frequencias.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pasta", default=AQUI / "outputs", help="Pasta com as subpastas diaria/ e semanal/.")
    args = parser.parse_args()
    pasta = Path(args.pasta)

    todas = ler_rodadas(pasta)
    resumo = melhores(todas)
    cols = ["frequencia", "h", "h_semanas", "modelo", "familia", "MAE", "MAPE_percent", "MAE_rel_naive",
            "RMSE_rel_naive", "acerto_direcao_percent", "dm_pvalor", "vs_naive"]
    todas[[c for c in cols if c in todas]].sort_values(["h_semanas", "frequencia", "MAE_rel_naive"]).to_csv(
        pasta / "comparacao_frequencias.csv", index=False, float_format="%.4f")
    grafico(resumo, pasta)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 180):
        print(resumo.drop(columns="n_modelos").to_string(index=False))
    print(f"\nSaidas: {(pasta / 'comparacao_frequencias.csv').resolve()} e comparacao_frequencias.png")


if __name__ == "__main__":
    main()

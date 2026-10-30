"""Baixa o IPCA mensal (IBGE) da API publica do Banco Central (SGS, serie 433) e grava
dados/ipca_mensal.csv com as colunas data (1o dia do mes) e variacao_percent (variacao no mes, %).

Uso: .venv/bin/python dados/baixar_ipca.py      (ou: make ipca)
"""
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
       "?formato=json&dataInicial=01/01/{ini}&dataFinal=31/12/{fim}")


def baixar(ano_ini=1999, ano_fim=None):
    ano_fim = ano_fim or date.today().year
    partes = []
    for ini in range(ano_ini, ano_fim + 1, 10):  # a API limita o tamanho de cada consulta
        fim = min(ini + 9, ano_fim)
        with urllib.request.urlopen(URL.format(ini=ini, fim=fim), timeout=60) as r:
            partes += json.load(r)
    df = pd.DataFrame(partes)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["variacao_percent"] = df["valor"].astype(float)
    return df[["data", "variacao_percent"]].drop_duplicates("data").sort_values("data")


if __name__ == "__main__":
    ipca = baixar()
    ipca.to_csv(AQUI / "ipca_mensal.csv", index=False)
    print(f"{len(ipca)} meses, de {ipca.data.min():%m/%Y} a {ipca.data.max():%m/%Y} -> {AQUI / 'ipca_mensal.csv'}")

"""Extrai o indicador da soja CEPEA/ESALQ - Parana (R$/saca 60 kg, a vista) da planilha
baixada do site do CEPEA (consulta ao banco de dados) e gera, nesta pasta:

  soja_diario.csv   data, preco_rs, preco_usd (um dia util com cotacao por linha)
  soja_semanal.csv  media semanal (semana terminando na sexta, como o milho)

A planilha do CEPEA vem com um defeito na estrutura do arquivo que faz o pandas recusar a
leitura ("Workbook corruption"); o xlrd le normalmente com ignore_workbook_corruption=True.

Uso:
    python dados_soja/extrair_soja.py [planilha.xls]   # padrao: a cepea-*.xls mais recente desta pasta
"""
import sys
from pathlib import Path

import pandas as pd
import xlrd

AQUI = Path(__file__).resolve().parent


def numero(v):
    """Celulas vem como texto com virgula decimal ("32,43") ou ja como numero."""
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip().replace(".", "").replace(",", ".")
    return float(v) if v else float("nan")


def carregar_soja(arquivo):
    aba = xlrd.open_workbook(arquivo, ignore_workbook_corruption=True).sheet_by_index(0)
    linhas = []
    for i in range(aba.nrows):
        data, rs, usd = (aba.cell_value(i, j) for j in range(3))
        try:
            dia = pd.to_datetime(str(data).strip(), format="%d/%m/%Y")
        except ValueError:  # cabecalho (titulo, nota, fonte, nomes das colunas)
            continue
        linhas.append({"data": dia, "preco_rs": numero(rs), "preco_usd": numero(usd)})
    return pd.DataFrame(linhas).drop_duplicates("data").set_index("data").sort_index()


def semanal(diario):
    """Media semanal (semana terminando na sexta). Semanas sem cotacao ficam NaN."""
    return diario.resample("W-FRI").mean()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        planilha = Path(sys.argv[1])
    else:
        candidatas = sorted(AQUI.glob("cepea-*.xls"))
        if not candidatas:
            raise SystemExit(f"Nenhuma planilha cepea-*.xls em {AQUI}")
        planilha = candidatas[-1]
    diario = carregar_soja(planilha)
    print(f"{planilha.name}: {len(diario)} dias, {diario.index.min().date()} a {diario.index.max().date()}")
    diario.to_csv(AQUI / "soja_diario.csv", float_format="%.2f")
    semanal(diario).to_csv(AQUI / "soja_semanal.csv", float_format="%.2f")
    print(f"Gravados {AQUI / 'soja_diario.csv'} e {AQUI / 'soja_semanal.csv'}")

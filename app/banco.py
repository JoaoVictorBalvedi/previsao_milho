"""Banco de dados do app: series de preco (milho e soja), execucoes e seus resultados.

DATABASE_URL escolhe o banco (no docker compose: PostgreSQL). Sem ela, usa um arquivo
SQLite em app/previsao.db, para rodar fora do Docker.

Tabelas:
  precos      uma linha por (produto, frequencia, data, coluna): a serie e as pracas
  execucoes   cada pedido de previsao: parametros, chave de reaproveitamento, status e log
  previsoes   previsoes do periodo de teste (previsoes.csv da v5)
  metricas    metricas por modelo (metricas.csv)
  futuro      previsao depois da ultima data (previsao_futura.csv)
  graficos    os PNGs gerados pela v5
"""
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (Column, Date, DateTime, Float, Integer, LargeBinary, MetaData, String, Table, Text,
                        create_engine, delete, func, select)

AQUI = Path(__file__).resolve().parent
URL = os.environ.get("DATABASE_URL") or f"sqlite:///{AQUI / 'previsao.db'}"
engine = create_engine(URL, pool_pre_ping=True)
meta = MetaData()

precos = Table(
    "precos", meta,
    Column("produto", String(20), primary_key=True),
    Column("frequencia", String(20), primary_key=True),
    Column("data", Date, primary_key=True),
    Column("coluna", String(60), primary_key=True),
    Column("ordem", Integer, nullable=False),  # ordem da coluna no CSV (define aux_1, aux_2... do professor)
    Column("valor", Float),
)

execucoes = Table(
    "execucoes", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chave", String(64), index=True, nullable=False),
    Column("parametros", Text, nullable=False),  # JSON normalizado (ver execucao.normalizar)
    Column("versao_dados", String(20), nullable=False),
    Column("status", String(20), nullable=False),  # na_fila, rodando, concluida, erro, cancelada
    Column("criada_em", DateTime, nullable=False),
    Column("iniciada_em", DateTime),
    Column("terminada_em", DateTime),
    Column("log", Text),
    Column("erro", Text),
)

previsoes = Table(
    "previsoes", meta,
    Column("execucao_id", Integer, index=True, nullable=False),
    Column("modelo", String(40), nullable=False),
    Column("origem", Integer), Column("alvo", Integer),
    Column("data_origem", Date), Column("data_alvo", Date),
    Column("pred", Float), Column("q10", Float), Column("q90", Float),
    Column("y_origem", Float), Column("real", Float),
)

metricas = Table(
    "metricas", meta,
    Column("execucao_id", Integer, index=True, nullable=False),
    Column("modelo", String(40), nullable=False), Column("familia", String(20)), Column("n", Integer),
    Column("mae", Float), Column("mse", Float), Column("rmse", Float), Column("mape_percent", Float),
    Column("mae_rel_naive", Float), Column("rmse_rel_naive", Float), Column("acerto_direcao_percent", Float),
    Column("dm_pvalor", Float), Column("vs_naive", String(30)), Column("cobertura_80_percent", Float),
)

futuro = Table(
    "futuro", meta,
    Column("execucao_id", Integer, index=True, nullable=False),
    Column("modelo", String(40), nullable=False), Column("passo", Integer), Column("data", Date),
    Column("pred", Float), Column("q10", Float), Column("q90", Float), Column("variacao_percent", Float),
)

graficos = Table(
    "graficos", meta,
    Column("execucao_id", Integer, index=True, nullable=False),
    Column("nome", String(80), nullable=False),
    Column("png", LargeBinary, nullable=False),
)

RESULTADOS = {"previsoes": previsoes, "metricas": metricas, "futuro": futuro}


def criar_tabelas():
    meta.create_all(engine)


def registros(df, tabela):
    """DataFrame -> lista de dicts so com as colunas da tabela, NaN virando NULL e datas como date."""
    cols = [c.name for c in tabela.columns if c.name in df.columns]
    df = df[cols].copy()
    for c in cols:
        if isinstance(tabela.c[c].type, Date):
            df[c] = pd.to_datetime(df[c]).dt.date
    df = df.astype(object).where(df.notna(), None)
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()} for r in df.to_dict("records")]


def inserir(conn, tabela, df, lote=5000):
    linhas = registros(df, tabela)
    for i in range(0, len(linhas), lote):
        conn.execute(tabela.insert(), linhas[i: i + lote])


# ============================================================
# SERIES DE PRECO
# ============================================================

def importar_serie(produto, frequencia, arquivo):
    """Substitui a serie (produto, frequencia) pelo conteudo de um CSV no formato da v5."""
    df = pd.read_csv(arquivo, parse_dates=["data"])
    ordem = {c: i for i, c in enumerate(df.columns) if c != "data"}
    longo = df.melt(id_vars="data", var_name="coluna", value_name="valor")
    longo["ordem"] = longo.coluna.map(ordem)
    longo["produto"], longo["frequencia"] = produto, frequencia
    with engine.begin() as conn:
        conn.execute(delete(precos).where(precos.c.produto == produto, precos.c.frequencia == frequencia))
        inserir(conn, precos, longo)
    _exportados.pop((produto, frequencia), None)
    return len(df)


def tem_serie(produto, frequencia):
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(precos).where(
            precos.c.produto == produto, precos.c.frequencia == frequencia)).scalar() > 0


def ler_serie(produto, frequencia):
    """A serie em formato largo (como o CSV original), com as colunas na ordem original."""
    with engine.connect() as conn:
        longo = pd.read_sql(select(precos.c.data, precos.c.coluna, precos.c.ordem, precos.c.valor).where(
            precos.c.produto == produto, precos.c.frequencia == frequencia), conn)
    if longo.empty:
        return None
    ordem = longo.drop_duplicates("coluna").sort_values("ordem").coluna.tolist()
    largo = longo.pivot(index="data", columns="coluna", values="valor")[ordem].sort_index()
    largo.index = pd.to_datetime(largo.index)
    largo.index.name = "data"
    return largo


_exportados = {}  # (produto, frequencia) -> (csv, versao); limpo quando a serie e reimportada


def exportar_csv(produto, frequencia):
    """(bytes do CSV para a v5, versao dos dados = hash do conteudo)."""
    if (produto, frequencia) not in _exportados:
        largo = ler_serie(produto, frequencia)
        if largo is None:
            raise ValueError(f"Nao ha dados de {produto} ({frequencia}) no banco.")
        buf = io.StringIO()
        largo.to_csv(buf, date_format="%Y-%m-%d")
        conteudo = buf.getvalue().encode()
        _exportados[(produto, frequencia)] = (conteudo, hashlib.sha1(conteudo).hexdigest()[:12])
    return _exportados[(produto, frequencia)]


def resumo_series():
    with engine.connect() as conn:
        r = conn.execute(select(precos.c.produto, precos.c.frequencia, func.min(precos.c.data),
                                func.max(precos.c.data), func.count(func.distinct(precos.c.data)))
                         .group_by(precos.c.produto, precos.c.frequencia)).all()
    return [{"produto": p, "frequencia": f, "inicio": str(a), "fim": str(b), "linhas": n} for p, f, a, b, n in r]

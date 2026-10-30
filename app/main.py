"""Interface web das previsoes (v5) com banco de dados.

    uvicorn app.main:app --port 8000        # fora do Docker (SQLite em app/previsao.db)
    docker compose up --build              # com PostgreSQL; abre em http://localhost:8000

Na primeira vez, as series de milho e soja sao copiadas dos CSVs para o banco. Depois o
banco e a fonte: as previsoes usam a serie do banco, e uma planilha nova do CEPEA pode ser
enviada pela propria interface.
"""
import importlib.util
import io
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import delete, select

from . import banco
from .execucao import RAIZ, TODOS, ParametroInvalido, apagar_resultados, bt, fila, linha_execucao, pedir, v5

ESTATICOS = Path(__file__).resolve().parent / "static"
PRODUTOS = {
    "milho": {"nome": "Milho", "fonte": "SIMA/DERAL, média das praças do Paraná", "coluna": "media_pr"},
    "soja": {"nome": "Soja", "fonte": "Indicador CEPEA/ESALQ Paraná, à vista", "coluna": "preco_rs"},
}
# textos da interface (os scripts usam ASCII)
NOME_FREQ = {"diaria": "Diária", "semanal": "Semanal"}
PLURAL = {"diaria": "dias úteis", "semanal": "semanas"}
NOME_FAMILIA = {"BASE": "Referências", "PROF": "Professor (prevê o preço)", "PROFVAR": "Professor prevendo a variação",
                "RET": "Retorno (direto, um modelo por horizonte)", "CHRONOS": "Chronos (pré-treinado, sem treino)"}


def caminho_csv(produto, frequencia):
    try:
        return v5.arquivo_dados(produto, v5.FREQUENCIAS[frequencia])
    except SystemExit:
        return None


def importar_csvs(so_vazias=True):
    feitos = []
    for produto in PRODUTOS:
        for frequencia in v5.FREQUENCIAS:
            arq = caminho_csv(produto, frequencia)
            if arq and not (so_vazias and banco.tem_serie(produto, frequencia)):
                n = banco.importar_serie(produto, frequencia, arq)
                feitos.append({"produto": produto, "frequencia": frequencia, "linhas": n, "arquivo": arq.name})
    return feitos


@asynccontextmanager
async def ciclo(app):
    banco.criar_tabelas()
    for f in importar_csvs(so_vazias=True):
        print(f"Importado para o banco: {f['produto']} {f['frequencia']} ({f['linhas']} linhas de {f['arquivo']})")
    fila.iniciar()
    yield


app = FastAPI(title="Previsao de precos - milho e soja (PR)", lifespan=ciclo)
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


@app.get("/")
def inicio():
    return FileResponse(ESTATICOS / "index.html")


def lista(v):
    """Array numpy/Series -> lista JSON (NaN -> null, numeros arredondados)."""
    return [None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else round(float(x), 4)
            for x in v]


# ============================================================
# OPCOES E DADOS
# ============================================================

@app.get("/api/opcoes")
def opcoes():
    familias = []
    for f in bt.FAMILIAS:
        if f == "COMBO":
            continue
        modelos = [m for m in TODOS if bt.familia(m) == f]
        familias.append({"id": f, "nome": NOME_FAMILIA.get(f, bt.NOME_FAMILIA[f]), "cor": bt.COR_FAMILIA[f],
                         "modelos": modelos})
    return {
        "produtos": [{"id": k, **v} for k, v in PRODUTOS.items()],
        "frequencias": [{"id": k, "nome": NOME_FREQ[k], "unidade": c["unidade"], "plural": PLURAL[k],
                         "max_horizonte": c["por_ano"]} for k in ("diaria", "semanal")
                        for c in [v5.FREQUENCIAS[k]]],
        "familias": familias,
        "combos": {k: v[1] for k, v in bt.COMBOS.items()},
        "cor_familia": {**bt.COR_FAMILIA, "NAIVE": bt.COR_NAIVE},
        "series": banco.resumo_series(),
    }


@app.get("/api/serie")
def serie(produto: str, frequencia: str):
    largo = banco.ler_serie(produto, frequencia)
    if largo is None or produto not in PRODUTOS:
        raise HTTPException(404, "Serie nao encontrada no banco")
    y = largo[PRODUTOS[produto]["coluna"]]
    return {"datas": [d.strftime("%Y-%m-%d") for d in largo.index], "valores": lista(y)}


@app.post("/api/dados/reimportar")
def reimportar():
    """Recarrega no banco todas as series a partir dos CSVs (ex.: depois de make dados-soja)."""
    return {"importadas": importar_csvs(so_vazias=False), "series": banco.resumo_series()}


@app.post("/api/dados/soja")
async def enviar_planilha_soja(arquivo: UploadFile = File(...)):
    """Recebe a planilha baixada do CEPEA, extrai a serie diaria e semanal, grava os CSVs em
    dados_soja/ e atualiza o banco. Previsoes antigas da soja deixam de ser reaproveitadas
    (a versao dos dados muda)."""
    spec = importlib.util.spec_from_file_location("extrair_soja", RAIZ / "dados_soja" / "extrair_soja.py")
    extrair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extrair)
    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "planilha.xls"
        caminho.write_bytes(await arquivo.read())
        try:
            diario = extrair.carregar_soja(caminho)
        except Exception as e:
            raise HTTPException(400, f"Nao consegui ler a planilha do CEPEA: {e}")
    if diario.empty:
        raise HTTPException(400, "A planilha nao tem linhas de preco (data dd/mm/aaaa, R$, US$).")
    pasta = RAIZ / "dados_soja"
    diario.to_csv(pasta / "soja_diario.csv", float_format="%.2f")
    extrair.semanal(diario).to_csv(pasta / "soja_semanal.csv", float_format="%.2f")
    for frequencia, nome in (("diaria", "soja_diario.csv"), ("semanal", "soja_semanal.csv")):
        banco.importar_serie("soja", frequencia, pasta / nome)
    return {"dias": len(diario), "inicio": str(diario.index.min().date()), "fim": str(diario.index.max().date()),
            "series": banco.resumo_series()}


# ============================================================
# PREVISOES
# ============================================================

class Pedido(BaseModel):
    produto: str = "milho"
    frequencia: str = "diaria"
    modo: str = "passo"
    horizonte: int | None = 5
    treino: float = 0.8
    janela_anos: int | None = None
    entrada: int | None = None
    modelos: list[str] = ["ALL"]
    forcar: bool = False


@app.post("/api/previsoes")
def prever(p: Pedido):
    """Reaproveita a mesma previsao se ja existir (forcar=false) ou entra na fila para treinar."""
    try:
        ex, reaproveitada = pedir(p.model_dump(exclude={"forcar"}), forcar=p.forcar)
    except ParametroInvalido as e:
        raise HTTPException(400, str(e))
    return {"execucao": resumo_execucao(ex), "reaproveitada": reaproveitada}


def resumo_execucao(ex, com_log=False):
    d = {k: v for k, v in ex.items() if k not in ("log", "chave")}
    if com_log:
        d["log"] = "\n".join((ex.get("log") or "").splitlines()[-80:])
    return d


def buscar(execucao_id):
    with banco.engine.connect() as conn:
        r = conn.execute(select(banco.execucoes).where(banco.execucoes.c.id == execucao_id)).first()
    if r is None:
        raise HTTPException(404, "Execucao nao encontrada")
    return linha_execucao(r)


@app.get("/api/execucoes")
def execucoes(limite: int = 60):
    with banco.engine.connect() as conn:
        rs = conn.execute(select(banco.execucoes).order_by(banco.execucoes.c.id.desc()).limit(limite)).all()
        melhores = dict(conn.execute(select(banco.metricas.c.execucao_id, banco.metricas.c.modelo)
                                     .where(banco.metricas.c.execucao_id.in_([r.id for r in rs]))
                                     .order_by(banco.metricas.c.mae_rel_naive.desc())).all())
    saida = []
    for r in rs:
        d = resumo_execucao(linha_execucao(r))
        d["melhor_modelo"] = melhores.get(r.id)  # a ultima linha de cada execucao (menor erro) fica no dict
        saida.append(d)
    return saida


@app.get("/api/execucoes/{execucao_id}")
def execucao(execucao_id: int):
    ex = buscar(execucao_id)
    d = resumo_execucao(ex, com_log=True)
    d["na_frente"] = 0
    if ex["status"] == "na_fila":
        with banco.engine.connect() as conn:
            d["na_frente"] = len(conn.execute(select(banco.execucoes.c.id).where(
                banco.execucoes.c.status.in_(["na_fila", "rodando"]), banco.execucoes.c.id < execucao_id)).all())
    return d


@app.post("/api/execucoes/{execucao_id}/cancelar")
def cancelar(execucao_id: int):
    buscar(execucao_id)
    if not fila.cancelar(execucao_id):
        raise HTTPException(409, "A execucao nao esta na fila nem rodando")
    return {"ok": True}


@app.delete("/api/execucoes/{execucao_id}")
def apagar(execucao_id: int):
    ex = buscar(execucao_id)
    if ex["status"] in ("na_fila", "rodando"):
        raise HTTPException(409, "Cancele a execucao antes de apagar")
    with banco.engine.begin() as conn:
        apagar_resultados(conn, execucao_id)
        conn.execute(delete(banco.execucoes).where(banco.execucoes.c.id == execucao_id))
    return {"ok": True}


def ler_tabela(nome, execucao_id):
    t = banco.RESULTADOS[nome]
    with banco.engine.connect() as conn:
        return pd.read_sql(select(t).where(t.c.execucao_id == execucao_id), conn).drop(columns="execucao_id")


@app.get("/api/execucoes/{execucao_id}/resultado")
def resultado(execucao_id: int):
    ex = buscar(execucao_id)
    if ex["status"] != "concluida":
        raise HTTPException(409, "A execucao ainda nao terminou")
    p = ex["parametros"]
    met = ler_tabela("metricas", execucao_id).sort_values("mae_rel_naive")
    prev = ler_tabela("previsoes", execucao_id)
    fut = ler_tabela("futuro", execucao_id)

    piv = prev.pivot_table(index="data_alvo", columns="modelo", values="pred", aggfunc="first").sort_index()
    real = prev.drop_duplicates("data_alvo").set_index("data_alvo").real.reindex(piv.index)
    teste = {"datas": [str(d) for d in piv.index], "real": lista(real), "modelos": {}}
    for m in piv.columns:
        teste["modelos"][m] = {"pred": lista(piv[m])}
        q = prev[prev.modelo == m].set_index("data_alvo").reindex(piv.index)
        if q.q10.notna().any():
            teste["modelos"][m].update(q10=lista(q.q10), q90=lista(q.q90))

    futuro = None
    if not fut.empty:
        datas = sorted(fut.data.unique())
        futuro = {"datas": [str(d) for d in datas], "modelos": {}}
        for m, g in fut.groupby("modelo"):
            g = g.set_index("data").reindex(datas)
            futuro["modelos"][m] = {"pred": lista(g.pred), "q10": lista(g.q10), "q90": lista(g.q90)}
        final = fut[fut.passo == fut.passo.max()].merge(met[["modelo", "mae_rel_naive", "vs_naive"]], on="modelo",
                                                        how="left").sort_values("mae_rel_naive")
        futuro["final"] = json.loads(final.assign(data=final.data.astype(str)).to_json(orient="records"))

    largo = banco.ler_serie(p["produto"], p["frequencia"])
    y = largo[PRODUTOS[p["produto"]]["coluna"]]
    with banco.engine.connect() as conn:
        nomes = [r.nome for r in conn.execute(select(banco.graficos.c.nome).where(
            banco.graficos.c.execucao_id == execucao_id).order_by(banco.graficos.c.nome))]
    return {
        "execucao": resumo_execucao(ex),
        "metricas": json.loads(met.to_json(orient="records")),
        "teste": teste,
        "futuro": futuro,
        "historico": {"datas": [d.strftime("%Y-%m-%d") for d in y.index], "valores": lista(y)},
        "graficos": nomes,
    }


@app.get("/api/execucoes/{execucao_id}/graficos/{nome}.png")
def grafico(execucao_id: int, nome: str):
    with banco.engine.connect() as conn:
        png = conn.execute(select(banco.graficos.c.png).where(
            banco.graficos.c.execucao_id == execucao_id, banco.graficos.c.nome == nome)).scalar()
    if png is None:
        raise HTTPException(404, "Grafico nao encontrado")
    return Response(png, media_type="image/png")


@app.get("/api/execucoes/{execucao_id}/csv/{tabela}")
def baixar_csv(execucao_id: int, tabela: str):
    if tabela not in banco.RESULTADOS:
        raise HTTPException(404, "Tabela desconhecida")
    df = ler_tabela(tabela, execucao_id)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{tabela}_{execucao_id}.csv"'})

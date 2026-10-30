"""Pedidos de previsao: normaliza os parametros, decide se da para reaproveitar uma
execucao anterior e roda a v5 (v5_diario_semanal/previsao.py) numa fila, um treino por vez.

A v5 roda como subprocesso, do mesmo jeito que na linha de comando: recebe a serie
exportada do banco (--data-file) e grava os CSVs/PNGs numa pasta temporaria, que depois
vao para o banco. Assim o script continua funcionando sozinho e o app so o envolve.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert, select, update

from . import banco

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = RAIZ / "v5_diario_semanal" / "previsao.py"
sys.path.insert(0, str(RAIZ / "v5_diario_semanal"))
import previsao as v5  # noqa: E402  (PRODUTOS, FREQUENCIAS e, via v5.bt, o catalogo de modelos)

bt = v5.bt
CHRONOS_IDS = ["amazon/chronos-bolt-small", "amazon/chronos-2"]
TODOS = bt.catalogo(CHRONOS_IDS)
LINHAS_LOG = 400  # o log guardado fica com as ultimas linhas


class ParametroInvalido(ValueError):
    pass


def normalizar(p):
    """Parametros do formulario -> forma canonica (mesmas escolhas = mesmo JSON = mesma chave)."""
    produto = p.get("produto", "milho")
    frequencia = p.get("frequencia", "diaria")
    modo = p.get("modo", "passo")
    if produto not in v5.PRODUTOS:
        raise ParametroInvalido(f"Produto desconhecido: {produto}")
    if frequencia not in v5.FREQUENCIAS:
        raise ParametroInvalido(f"Frequencia desconhecida: {frequencia}")
    if modo not in ("passo", "cego"):
        raise ParametroInvalido(f"Modo desconhecido: {modo}")
    cfg = v5.FREQUENCIAS[frequencia]

    def inteiro(nome, minimo, maximo=None):
        v = p.get(nome)
        if v in (None, ""):
            return None
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ParametroInvalido(f"{nome} deve ser um numero inteiro")
        if v < minimo or (maximo is not None and v > maximo):
            raise ParametroInvalido(f"{nome} deve estar entre {minimo} e {maximo}" if maximo else
                                    f"{nome} deve ser pelo menos {minimo}")
        return v

    horizonte = None
    if modo == "passo":
        horizonte = inteiro("horizonte", 1, cfg["por_ano"])
        if horizonte is None:
            raise ParametroInvalido("Informe o horizonte")
    treino = round(float(p.get("treino", 0.8)), 2)
    if not 0.5 <= treino <= 0.95:
        raise ParametroInvalido("treino deve estar entre 0.5 e 0.95")
    pedidos = p.get("modelos") or ["ALL"]
    try:
        modelos = bt.selecionar_modelos(",".join(pedidos), TODOS)
    except SystemExit as e:  # selecionar_modelos sai com mensagem quando o nome nao existe
        raise ParametroInvalido(str(e))
    if modo == "cego":
        modelos = [m for m in modelos if bt.familia(m) != "RET"]
    modelos = [m for m in modelos if bt.familia(m) != "COMBO"]  # combinacoes saem sozinhas
    return {"produto": produto, "frequencia": frequencia, "modo": modo, "horizonte": horizonte,
            "treino": treino, "janela_anos": inteiro("janela_anos", 1, 30), "entrada": inteiro("entrada", 1, 2000),
            "modelos": modelos}


def chave(params, versao_dados):
    texto = json.dumps({**params, "dados": versao_dados}, sort_keys=True)
    return hashlib.sha1(texto.encode()).hexdigest()


def linha_execucao(r):
    d = dict(r._mapping)
    d["parametros"] = json.loads(d["parametros"])
    for c in ("criada_em", "iniciada_em", "terminada_em"):
        d[c] = d[c].isoformat() if d[c] else None
    return d


def pedir(params_brutos, forcar=False):
    """Devolve (execucao, reaproveitada). Sem forcar, reaproveita a ultima execucao com a
    mesma chave que terminou bem ou ainda esta na fila/rodando."""
    params = normalizar(params_brutos)
    if not banco.tem_serie(params["produto"], params["frequencia"]):
        raise ParametroInvalido(f"Nao ha dados de {params['produto']} ({params['frequencia']}) no banco.")
    _, versao = banco.exportar_csv(params["produto"], params["frequencia"])
    k = chave(params, versao)
    with banco.engine.begin() as conn:
        if not forcar:
            r = conn.execute(select(banco.execucoes).where(
                banco.execucoes.c.chave == k, banco.execucoes.c.status.in_(["concluida", "na_fila", "rodando"]))
                .order_by(banco.execucoes.c.id.desc()).limit(1)).first()
            if r:
                return linha_execucao(r), True
        novo = conn.execute(insert(banco.execucoes).values(
            chave=k, parametros=json.dumps(params), versao_dados=versao, status="na_fila",
            criada_em=datetime.now(), log="")).inserted_primary_key[0]
        r = conn.execute(select(banco.execucoes).where(banco.execucoes.c.id == novo)).first()
    fila.acordar()
    return linha_execucao(r), False


def comando(params, arquivo_dados, pasta_saida):
    cmd = [sys.executable, "-u", str(SCRIPT), "--produto", params["produto"], "--frequencia", params["frequencia"],
           "--treino", str(params["treino"]), "--models", ",".join(params["modelos"]),
           "--data-file", str(arquivo_dados), "--output-dir", str(pasta_saida), "--limpar"]
    cmd += ["--modo", "cego"] if params["modo"] == "cego" else ["--horizonte", str(params["horizonte"])]
    if params["janela_anos"]:
        cmd += ["--janela-anos", str(params["janela_anos"])]
    if params["entrada"]:
        cmd += ["--entrada", str(params["entrada"])]
    return cmd


def salvar_resultados(conn, execucao_id, pasta):
    prev = pd.read_csv(pasta / "previsoes.csv")
    met = pd.read_csv(pasta / "metricas.csv").rename(columns=str.lower)
    banco.inserir(conn, banco.previsoes, prev.assign(execucao_id=execucao_id))
    banco.inserir(conn, banco.metricas, met.assign(execucao_id=execucao_id))
    if (pasta / "previsao_futura.csv").exists():
        fut = pd.read_csv(pasta / "previsao_futura.csv")
        banco.inserir(conn, banco.futuro, fut.assign(execucao_id=execucao_id))
    for png in sorted(pasta.glob("*.png")):
        conn.execute(insert(banco.graficos).values(execucao_id=execucao_id, nome=png.stem, png=png.read_bytes()))


def apagar_resultados(conn, execucao_id):
    for t in (banco.previsoes, banco.metricas, banco.futuro, banco.graficos):
        conn.execute(delete(t).where(t.c.execucao_id == execucao_id))


class Fila:
    """Uma thread que pega a execucao mais antiga na fila e roda a v5. Um treino por vez:
    os modelos ja usam todos os nucleos e o Chronos ocupa bastante memoria."""

    def __init__(self):
        self.evento = threading.Event()
        self.processo = None
        self.atual = None
        self.cancelar_atual = False
        self.thread = None

    def iniciar(self):
        with banco.engine.begin() as conn:  # o servidor caiu no meio de um treino
            conn.execute(update(banco.execucoes).where(banco.execucoes.c.status == "rodando").values(
                status="erro", erro="Interrompida: o servidor foi reiniciado durante o treino.",
                terminada_em=datetime.now()))
        self.thread = threading.Thread(target=self.laco, daemon=True, name="fila-previsoes")
        self.thread.start()

    def acordar(self):
        self.evento.set()

    def laco(self):
        while True:
            with banco.engine.connect() as conn:
                r = conn.execute(select(banco.execucoes).where(banco.execucoes.c.status == "na_fila")
                                 .order_by(banco.execucoes.c.id).limit(1)).first()
            if r is None:
                self.evento.wait(5)
                self.evento.clear()
                continue
            try:
                self.rodar(linha_execucao(r))
            except Exception as e:  # nunca derrubar a thread da fila
                self.terminar(r.id, "erro", f"{type(e).__name__}: {e}")

    def atualizar(self, execucao_id, **valores):
        with banco.engine.begin() as conn:
            conn.execute(update(banco.execucoes).where(banco.execucoes.c.id == execucao_id).values(**valores))

    def terminar(self, execucao_id, status, erro=None):
        self.atualizar(execucao_id, status=status, erro=erro, terminada_em=datetime.now())

    def rodar(self, ex):
        params = ex["parametros"]
        self.atual, self.cancelar_atual = ex["id"], False
        self.atualizar(ex["id"], status="rodando", iniciada_em=datetime.now(), log="")
        with tempfile.TemporaryDirectory(prefix="previsao_") as tmp:
            tmp = Path(tmp)
            conteudo, _ = banco.exportar_csv(params["produto"], params["frequencia"])
            (tmp / "dados.csv").write_bytes(conteudo)
            saida = tmp / "saida"
            env = {**os.environ, "PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8",
                   "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "TQDM_DISABLE": "1"}
            extra = {"start_new_session": True} if os.name != "nt" else \
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            self.processo = subprocess.Popen(comando(params, tmp / "dados.csv", saida), cwd=RAIZ, env=env,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                             encoding="utf-8", errors="replace", **extra)
            linhas, ultimo = [], 0.0
            for linha in self.processo.stdout:
                linha = linha.rstrip()
                if not linha or "it/s]" in linha or "%|" in linha:  # barras de progresso
                    continue
                linhas.append(linha)
                if time.monotonic() - ultimo > 1:
                    self.atualizar(ex["id"], log="\n".join(linhas[-LINHAS_LOG:]))
                    ultimo = time.monotonic()
            codigo = self.processo.wait()
            self.processo = None
            self.atualizar(ex["id"], log="\n".join(linhas[-LINHAS_LOG:]))
            if self.cancelar_atual:
                self.terminar(ex["id"], "cancelada")
            elif codigo != 0:
                ultimas = [l for l in linhas if l.strip()][-3:]
                self.terminar(ex["id"], "erro", "O treino terminou com erro: " + " | ".join(ultimas))
            else:
                with banco.engine.begin() as conn:
                    apagar_resultados(conn, ex["id"])
                    salvar_resultados(conn, ex["id"], saida)
                self.terminar(ex["id"], "concluida")
        self.atual = None

    def cancelar(self, execucao_id):
        """Na fila: so marca. Rodando: mata o processo e os filhos (o Chronos roda num filho)."""
        with banco.engine.begin() as conn:
            r = conn.execute(update(banco.execucoes).where(
                banco.execucoes.c.id == execucao_id, banco.execucoes.c.status == "na_fila")
                .values(status="cancelada", terminada_em=datetime.now()))
            if r.rowcount:
                return True
        if self.atual == execucao_id and self.processo is not None:
            self.cancelar_atual = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.processo.pid)], capture_output=True)
            else:
                os.killpg(self.processo.pid, signal.SIGKILL)
            return True
        return False


fila = Fila()

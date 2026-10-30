# Comandos do TCC. Rode sempre nesta pasta (a que tem este arquivo). Ex.: make holdout
# Opcoes podem ser trocadas na chamada: make holdout H=4 MODELOS=PROF TREINO=0.7

PY      := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)
H       ?= 1
FREQ    ?= diaria
PRODUTO ?= milho
MODELOS ?= ALL
TREINO  ?= 0.8
LIMPAR  ?=
JANELA  ?=
ENTRADA ?=
EXTRA   := $(if $(LIMPAR),--limpar,) $(if $(JANELA),--janela-anos $(JANELA),) $(if $(ENTRADA),--entrada $(ENTRADA),)
SUFIXO  := $(if $(filter-out milho,$(PRODUTO)),_$(PRODUTO),)$(if $(JANELA),_janela$(JANELA),)$(if $(ENTRADA),_entrada$(ENTRADA),)
comma   := ,
# no backtest, H so e repassado se voce escrever na chamada (ex.: H=2 ou H=1,4); senao roda 1,4,12,26
H_BACKTEST := $(if $(filter command line,$(origin H)),--horizons $(H),)

.PHONY: ajuda setup dados dados-soja app docker professor backtest backtest-rapido holdout holdout-cego holdout-todos v5 v5-cego v5-comparar reavaliar tudo

ajuda:  ## Lista os comandos
	@echo "Comandos disponiveis (rode na pasta do projeto):"
	@echo "  make setup            instala o ambiente (.venv com Python 3.12) - so na primeira vez"
	@echo "  make dados            regera dados_milho/milho_*.csv a partir das planilhas do SIMA"
	@echo "  make professor        roda o script original do professor (v2)"
	@echo "  make backtest         backtest com 39 pontos de teste em 2023-2025 (v3, ~7 min)"
	@echo "                        horizontes padrao 1,4,12,26; para escolher: make backtest H=2 (ou H=1,4)"
	@echo "  make backtest-rapido  o mesmo, com menos pontos e sem Chronos (~1 min)"
	@echo "  make holdout          treina com 80% e preve os 20% finais, 1 semana a frente (v4)"
	@echo "  make holdout-cego     treina com 80% e preve os 20% finais de uma vez so (v4)"
	@echo "  make holdout-todos    holdout em 1, 4 e 12 semanas + cego"
	@echo "  make docker           interface web + banco PostgreSQL (Docker): http://localhost:8000"
	@echo "  make app              interface web sem Docker (banco SQLite): http://localhost:8000"
	@echo "  make dados-soja       regera dados_soja/soja_*.csv a partir da planilha do CEPEA"
	@echo "  make v5               holdout escolhendo a frequencia: FREQ=diaria (padrao) ou FREQ=semanal (v5)"
	@echo "                        e o produto: PRODUTO=milho (padrao) ou PRODUTO=soja"
	@echo "                        H em dias uteis (1 a 240) ou semanas (1 a 52); varios: H=1,5,20"
	@echo "                        tambem preve os H passos apos a ultima data (previsao_futura.csv)"
	@echo "  make v5-cego          o mesmo, com uma previsao unica para os 20% finais"
	@echo "  make v5-comparar      compara diaria x semanal em todas as rodadas da v5 ja feitas"
	@echo "  make reavaliar        recalcula metricas, combinacoes e graficos sem rodar modelos (segundos)"
	@echo ""
	@echo "Rodar so alguns modelos (MODELOS=...) mantem os resultados dos outros; LIMPAR=1 comeca do zero."
	@echo "Opcoes: H=<semanas a frente>  MODELOS=<ALL|PROF|PROFVAR|RET|CHRONOS|nomes>  TREINO=<0.5 a 0.95>"
	@echo "        JANELA=<anos>  treina so com os ultimos N anos (resultados em outputs_janelaN/)"
	@echo "        ENTRADA=<semanas>  cada previsao so olha as ultimas N semanas (outputs_entradaN/)"
	@echo "Ex.:    make holdout ENTRADA=4 H=2   (olha 4 semanas para prever 2 a frente)"
	@echo "Ex.:    make holdout H=4 MODELOS=PROF_RIDGE,PROF_RF,NAIVE"

setup:
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt
	@echo "No Mac, o XGBoost/LightGBM precisam tambem de: brew install libomp"

dados:
	$(PY) dados_milho/extrair_milho.py

dados-soja:
	$(PY) dados_soja/extrair_soja.py

app:  ## interface web sem Docker (SQLite em app/previsao.db): http://localhost:8000
	$(PY) -m uvicorn app.main:app --port 8000

docker:  ## interface web + PostgreSQL no Docker: http://localhost:8000
	docker compose up --build

professor:
	$(PY) v2_modelo_professor/02_modelo_previsoes_milho.py \
		--data-file dados_milho/milho_semanal.csv --output-dir v2_modelo_professor/outputs

backtest:
	$(PY) v3_backtest/backtest.py --models $(MODELOS) $(H_BACKTEST) $(EXTRA)

backtest-rapido:
	$(PY) v3_backtest/backtest.py --step 13 --chronos-models "" --output-dir v3_backtest/outputs_rapido

holdout:
	$(PY) v4_holdout/holdout.py --horizonte $(H) --models $(MODELOS) --treino $(TREINO) $(EXTRA)

holdout-cego:
	$(PY) v4_holdout/holdout.py --modo cego --models $(MODELOS) --treino $(TREINO) $(EXTRA)

v5:
	@for h in $(subst $(comma), ,$(H)); do \
		$(PY) v5_diario_semanal/previsao.py --produto $(PRODUTO) --frequencia $(FREQ) --horizonte $$h --models $(MODELOS) \
			--treino $(TREINO) $(EXTRA) || exit 1; \
	done

v5-comparar:
	$(PY) v5_diario_semanal/comparar.py --pasta v5_diario_semanal/outputs$(SUFIXO)

v5-cego:
	$(PY) v5_diario_semanal/previsao.py --produto $(PRODUTO) --frequencia $(FREQ) --modo cego --models $(MODELOS) --treino $(TREINO) $(EXTRA)

holdout-todos:
	$(MAKE) holdout H=1
	$(MAKE) holdout H=4
	$(MAKE) holdout H=12
	$(MAKE) holdout-cego

reavaliar:
	@for d in v3_backtest/outputs*; do \
		test -f $$d/previsoes_backtest.csv && $(PY) v3_backtest/backtest.py --reavaliar --output-dir $$d; \
	done; true
	@for d in v4_holdout/outputs*/passo_h*; do \
		test -f $$d/previsoes.csv && $(PY) v4_holdout/holdout.py --horizonte $${d##*_h} --reavaliar --output-dir $$d; \
	done; true
	@for d in v4_holdout/outputs*/cego; do \
		test -f $$d/previsoes.csv && $(PY) v4_holdout/holdout.py --modo cego --reavaliar --output-dir $$d; \
	done; true
	@for d in v5_diario_semanal/outputs*/*/passo_h* v5_diario_semanal/outputs*/*/cego; do \
		f=$$(basename $$(dirname $$d)); p=milho; case $$d in *outputs_soja*) p=soja;; esac; \
		case $$d in *cego) modo="--modo cego";; *) modo="--horizonte $${d##*_h}";; esac; \
		test -f $$d/previsoes.csv && $(PY) v5_diario_semanal/previsao.py --produto $$p --frequencia $$f $$modo \
			--reavaliar --output-dir $$d; \
	done; true

tudo: backtest holdout-todos

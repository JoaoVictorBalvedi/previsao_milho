# Comandos do TCC. Rode sempre nesta pasta (a que tem este arquivo). Ex.: make holdout
# Opcoes podem ser trocadas na chamada: make holdout H=4 MODELOS=PROF TREINO=0.7

PY      := .venv/bin/python
H       ?= 1
MODELOS ?= ALL
TREINO  ?= 0.8
LIMPAR  ?=
JANELA  ?=
ENTRADA ?=
EXTRA   := $(if $(LIMPAR),--limpar,) $(if $(JANELA),--janela-anos $(JANELA),) $(if $(ENTRADA),--entrada $(ENTRADA),)
# no backtest, H so e repassado se voce escrever na chamada (ex.: H=2 ou H=1,4); senao roda 1,4,12,26
H_BACKTEST := $(if $(filter command line,$(origin H)),--horizons $(H),)

.PHONY: ajuda setup dados professor backtest backtest-rapido holdout holdout-cego holdout-todos reavaliar tudo

ajuda:  ## Lista os comandos
	@echo "Comandos disponiveis (rode na pasta do projeto):"
	@echo "  make setup            instala o ambiente (.venv com Python 3.12) - so na primeira vez"
	@echo "  make dados            regera dados/milho_semanal.csv a partir das planilhas do SIMA"
	@echo "  make professor        roda o script original do professor (v2)"
	@echo "  make backtest         backtest com 39 pontos de teste em 2023-2025 (v3, ~7 min)"
	@echo "                        horizontes padrao 1,4,12,26; para escolher: make backtest H=2 (ou H=1,4)"
	@echo "  make backtest-rapido  o mesmo, com menos pontos e sem Chronos (~1 min)"
	@echo "  make holdout          treina com 80% e preve os 20% finais, 1 semana a frente (v4)"
	@echo "  make holdout-cego     treina com 80% e preve os 20% finais de uma vez so (v4)"
	@echo "  make holdout-todos    holdout em 1, 4 e 12 semanas + cego"
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
	$(PY) dados/extrair_milho.py

professor:
	$(PY) v2_modelo_professor/02_modelo_previsoes_milho.py \
		--data-file dados/milho_semanal.csv --output-dir v2_modelo_professor/outputs

backtest:
	$(PY) v3_backtest/backtest.py --models $(MODELOS) $(H_BACKTEST) $(EXTRA)

backtest-rapido:
	$(PY) v3_backtest/backtest.py --step 13 --chronos-models "" --output-dir v3_backtest/outputs_rapido

holdout:
	$(PY) v4_holdout/holdout.py --horizonte $(H) --models $(MODELOS) --treino $(TREINO) $(EXTRA)

holdout-cego:
	$(PY) v4_holdout/holdout.py --modo cego --models $(MODELOS) --treino $(TREINO) $(EXTRA)

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

tudo: backtest holdout-todos

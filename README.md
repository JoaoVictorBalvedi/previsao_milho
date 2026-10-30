# TCC – Previsão do preço do milho (Paraná, SIMA/DERAL)

Preço semanal do milho (R$/saca 60 kg) nas praças do Paraná, de 2001 a 2025.
Alvo: `media_pr` (média simples das praças).

## Interface no navegador (Docker)

O jeito mais fácil de usar: precisa só do [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
docker compose up --build     # ou: make docker
```

Depois abra **http://localhost:8000**. A primeira vez demora alguns minutos, porque baixa o Python,
o torch e as bibliotecas. Nas próximas, `docker compose up` sobe em segundos. Para parar:
`docker compose down`. Os dados ficam guardados; `docker compose down -v` apaga o banco.

Na interface:

- **Todas as opções da v5:** produto (milho/soja), frequência (diária/semanal), teste passo a passo
  ou cego, horizonte, modelos (por família ou um a um), parte da série usada no treino, janela de
  treino e janela de entrada.
- **Previsão já feita não treina de novo.** Cada pedido gera uma chave a partir das opções e da
  versão dos dados. Se essa chave já foi calculada, o resultado salvo aparece na hora. O botão
  **Treinar de novo** ignora o que está salvo e treina outra vez. Se os dados mudarem (planilha nova
  da soja, por exemplo), a chave muda e o treino é refeito.
- **Treino em segundo plano.** Os pedidos entram numa fila e rodam um de cada vez. O andamento aparece ao vivo
  e pode ser cancelado. Dá para fechar a página e voltar depois: o **Histórico** guarda tudo.
- **Resultado:**
  - indicadores principais;
  - gráfico da previsão para os próximos dias/semanas;
  - gráfico real × previsto no período de teste;
  - tabela de métricas, onde a caixa de cada modelo liga e desliga o modelo nos gráficos;
  - os PNGs do script;
  - downloads em CSV.
- **Dados:** a planilha nova do CEPEA (soja) pode ser enviada pela própria página.

**Banco de dados (PostgreSQL).** Na primeira subida, as séries dos CSVs (`dados_milho/`
e `dados_soja/`) são copiadas para o banco. A partir daí, o banco é a fonte dos dados.

| Tabela | O que guarda |
|---|---|
| `precos` | as séries de milho e soja, diárias e semanais, uma linha por data e coluna |
| `execucoes` | cada pedido: opções, chave, status, log do treino e horários |
| `previsoes` | as previsões do período de teste de cada execução |
| `metricas` | as métricas por modelo |
| `futuro` | a previsão para depois da última data |
| `graficos` | os PNGs gerados pela v5 |

Para abrir o banco num cliente (DBeaver, pgAdmin), use `localhost:5433`, banco `previsao`, usuário e
senha `tcc`.

**Como funciona por dentro (`app/`).**

- `main.py` é a API (FastAPI).
- `banco.py` define as tabelas (SQLAlchemy).
- `execucao.py` cuida da fila e da chave de reaproveitamento.
- `static/` tem a página (HTML/CSS/JS, gráficos com Chart.js).

O treino continua sendo a v5 (`v5_diario_semanal/previsao.py`), chamada como subprocesso. A série
sai do banco para um CSV temporário e os resultados voltam para o banco. A linha de comando continua
funcionando igual.

**Sem Docker:** `make app` (ou `uvicorn app.main:app`) usa um arquivo SQLite (`app/previsao.db`) no
lugar do PostgreSQL.

## Passo a passo rápido

Abra o terminal **nesta pasta** (a que tem o `Makefile`) e rode:

```bash
make setup          # 1ª vez só: cria o ambiente Python e instala tudo (precisa do uv)
brew install libomp # 1ª vez só, no Mac: necessário para XGBoost e LightGBM

make holdout        # treina com 80% dos dados e prevê os 20% finais (1 semana à frente)
make backtest       # teste em 39 datas de 2023 a 2025, em 1, 4, 12 e 26 semanas
```

`make` sozinho (ou `make ajuda`) lista todos os comandos. Os resultados (tabelas e gráficos)
vão para a pasta `outputs/` de cada versão.

### Todos os comandos

| Comando | O que faz | Tempo |
|---|---|---|
| `make setup` | Cria `.venv` (Python 3.12) e instala `requirements.txt` | ~2 min |
| `make dados` | Regera `dados_milho/milho_semanal.csv` a partir das planilhas do SIMA | ~1 min |
| `make professor` | Roda o script original do professor (v2) | ~10 s |
| `make holdout` | Holdout 80/20, previsão 1 semana à frente (v4) | ~20 s |
| `make holdout H=4` | O mesmo, 4 semanas à frente (vale qualquer valor de 1 a 52) | ~2 min (H=12: ~5 min) |
| `make holdout-cego` | Holdout 80/20 com uma única previsão para os 20% inteiros | ~1 min |
| `make holdout-todos` | Holdout em 1, 4 e 12 semanas + cego | ~8 min |
| `make v5 H=5` | Holdout na série **diária** (ou `FREQ=semanal`), H passos à frente, + previsão futura (v5) | ~3 min sem Chronos, ~12 min com |
| `make v5-comparar` | Compara diária x semanal nas rodadas da v5 | segundos |
| `make backtest` | Backtest de 2023 a 2025 em 1, 4, 12 e 26 semanas (v3) | ~7 min |
| `make backtest-rapido` | Backtest com menos datas e sem Chronos | ~1 min |
| `make reavaliar` | Recalcula métricas, combinações e gráficos de todos os testes, sem rodar modelos | segundos |

Opções que funcionam nos comandos de holdout e backtest:

- `MODELOS=...` escolhe os modelos: `ALL` (padrão), uma família (`PROF`, `PROFVAR`, `RET`,
  `CHRONOS`, `BASE`) ou nomes separados por vírgula. O `NAIVE` sempre roda, porque é a referência.
  **Os modelos que não rodaram agora são mantidos da rodada anterior**, então dá para rodar só um
  modelo novo sem perder os outros.
- `LIMPAR=1` descarta os resultados anteriores e começa do zero.
- `TREINO=0.7` muda a fração usada no treino do holdout (padrão 0.8).
- `JANELA=10` usa uma **janela deslizante de treino**: o modelo estuda só os últimos 10 anos antes
  de cada previsão, em vez de todo o histórico desde 2001. Os resultados vão para pastas separadas
  (`v3_backtest/outputs_janela10/`, `v4_holdout/outputs_janela10/`), sem misturar com os normais.
- `ENTRADA=4` usa uma **janela de entrada**: em cada previsão, o modelo só enxerga os preços das
  últimas 4 semanas. Resultados em `outputs_entrada4/` (ou `outputs_janela10_entrada4/` se combinar).

```bash
make holdout MODELOS=PROF                 # roda só os 8 do professor (os outros continuam lá)
make holdout MODELOS=PROFVAR H=4          # acrescenta os PROFVAR ao holdout de 4 semanas
make holdout H=12 MODELOS=PROF_RIDGE,RET_RIDGE,CHRONOS_2
make holdout TREINO=0.7                   # treina com 70%, testa nos 30% finais
make holdout JANELA=10                    # holdout treinando só com 2011-2021
make backtest JANELA=10                   # backtest treinando só com os 10 anos antes de cada data
make holdout ENTRADA=4 H=2                # olha as últimas 4 semanas para prever 2 semanas à frente
make holdout ENTRADA=4 H=2 JANELA=10      # as três coisas juntas
make backtest MODELOS=PROF
```

Na primeira vez que um modelo Chronos roda, ele é baixado da internet (~200 MB).

## v5: previsão diária ou semanal

`make v5` faz o mesmo holdout 80/20 da v4, mas deixa escolher a série:

| Opção | Série | Um passo (`H`, `ENTRADA`) | `H` vai de |
|---|---|---|---|
| `FREQ=diaria` (padrão do `make v5`) | `dados_milho/milho_diario.csv`, preço de cada dia | 1 dia útil com cotação (~240 por ano) | 1 a 240 |
| `FREQ=semanal` | `dados_milho/milho_semanal.csv`, média da semana | 1 semana | 1 a 52 |

```bash
make v5 H=5                       # diária, 5 dias úteis (~1 semana) à frente
make v5 H=1,5,20,60               # vários horizontes numa chamada só
make v5 FREQ=semanal H=4          # semanal: dá o mesmo resultado que make holdout H=4, só que bem mais rápido
make v5-cego                      # diária, uma previsão única para os 20% finais
make v5 H=5 ENTRADA=20 JANELA=10  # olha os últimos 20 dias, treina com 10 anos
make v5-comparar                  # compara diária x semanal em todas as rodadas já feitas
```

Valem todas as opções da v4 (`MODELOS`, `LIMPAR`, `TREINO`, `JANELA`, `ENTRADA`). Resultados em
`v5_diario_semanal/outputs/<diaria|semanal>/passo_hNN/` (ou `cego/`), com os mesmos arquivos da v4.

**Como os modelos se adaptam à série diária.** As janelas cobrem o mesmo tempo de calendário que
na semanal (1 semana = 5 dias úteis, 1 mês = 20, 1 ano = 240):

- Professor (PROF_, PROFVAR_): lags de 1, 2, 3, 4, 5, 10, 20, 60, 120 e 240 dias e médias/desvios/mín/máx
  móveis de 5 a 240 dias. O `diff_52` do script dele continua sendo de 52 passos.
- Retorno (RET_): variações dos últimos 5 dias e de 10 e 20 dias atrás, variação acumulada em
  5/20/60/120/240 dias, volatilidade e distância da média; sazonalidade pelo dia do ano.
- SNAIVE: o preço do mesmo dia no ano anterior. O Chronos recebe a série diária sem mudança.

**Previsão recursiva mais rápida.** A previsão recursiva do professor recalcula as variáveis da série
inteira a cada passo. Na série diária, com ~1150 datas de teste, isso levaria horas. A v5 faz a mesma
conta para todas as datas de uma vez (`previsao_recursiva()`). Ela foi conferida contra o original:
as diferenças ficam abaixo de 1e-10. Na semanal a v5 reproduz a v4 e roda em segundos: 9 s, contra
217 s da v4 para H=4 com 7 modelos.

**Previsão para o futuro.** No modo passo, depois do holdout os modelos são retreinados com a série
inteira e preveem os H passos seguintes à última data. Saem `previsao_futura.csv` (com faixa de 80%
do Chronos e variação % sobre o último preço) e `06_previsao_futura.png`. A tabela mostra cada
previsão ao lado do erro do modelo no holdout, que diz em qual delas confiar. Para pular essa
etapa: `--sem-futuro`.

**Comparação diária x semanal (`make v5-comparar`).** O script junta as rodadas, converte o
horizonte para semanas (5 dias úteis = 1) e gera `comparacao_frequencias.csv` e `.png`. Os alvos não
são iguais: a semanal prevê a *média* da semana, que é mais suave. Por isso, para comparar as duas,
use o erro relativo ao ingênuo de cada uma, não o MAPE.

Primeiro teste (holdout, modelos BASE/RET/PROF_RIDGE/PROF_ELASTICNET), erro relativo ao ingênuo do
melhor modelo:

| Horizonte | Diária | Semanal |
|---|---|---|
| 1 dia | 0,93* (COMBO_MEDIANA) | — |
| ~1 semana | 0,93 (RET_EXTRATREES; H=5) | 0,89* (COMBO_MEDIANA; H=1) |
| ~1 mês | 0,97 (COMBO_NAIVE_RET; H=20) | 0,95 (COMBO_NAIVE_RET; H=4) |

\* diferença significativa (Diebold-Mariano, p < 0,05). Na diária com todos os modelos e H=5,
`COMBO_NAIVE_RET` (0,93) e `COMBO_LINEARES` (0,95) também ganham do ingênuo de forma significativa.
As árvores do professor falham na diária pelo mesmo motivo do holdout semanal (preço do teste acima
do máximo do treino). Os dois Chronos ficam atrás do ingênuo na diária: CHRONOS_BOLT_SMALL tem 1,05
e CHRONOS_2 tem 1,25 (pior, p < 0,05).

### Milho ou soja (`PRODUTO=soja`)

A v5 também treina e testa com o preço da **soja**: o indicador CEPEA/ESALQ Paraná à vista, em
R$/saca de 60 kg, diário de jan/2005 a out/2026.

```bash
make dados-soja                         # 1ª vez (ou ao baixar planilha nova): gera dados_soja/soja_*.csv
make v5 PRODUTO=soja H=5                # soja diária, 5 dias úteis à frente
make v5 PRODUTO=soja FREQ=semanal H=4
make v5-comparar PRODUTO=soja           # diária x semanal da soja
```

- A planilha baixada do CEPEA vai em `dados_soja/cepea-*.xls`. `dados_soja/extrair_soja.py` lê a
  mais recente e grava `soja_diario.csv` (data, `preco_rs`, `preco_usd`) e `soja_semanal.csv`
  (média da semana até a sexta, como o milho). As planilhas do CEPEA vêm com um defeito no arquivo
  que o pandas recusa; o script lê com o `xlrd` ignorando o defeito.
- O alvo é `preco_rs`. A soja não tem praças, então os modelos não têm as variáveis auxiliares
  (`aux_*` do professor, dispersão entre praças dos RET_). O preço em US$ fica no CSV, mas não
  entra nos modelos.
- O holdout 80/20 da soja treina de jan/2005 a jun/2022 e testa de jun/2022 a out/2026 (diária:
  4355 / 1089 dias).
- Resultados em `v5_diario_semanal/outputs_soja/` (e `outputs_soja_janela10/` etc.). Os do milho
  continuam em `outputs/`.

Primeiro teste com a soja (sem Chronos), erro relativo ao ingênuo:

| Modelo | Diária, H=5 | Semanal, H=4 |
|---|---|---|
| COMBO_NAIVE_RET | 0,98 | **0,93*** |
| RET_RIDGE | 0,99 | 0,96 |
| RET_EXTRATREES | 1,00 | 0,95 |
| PROF_RIDGE | 1,07 (pior*) | 1,18 (pior*) |
| CHRONOS_BOLT_SMALL | — | 1,02 |
| Árvores do professor (PROF_) | 3,1 a 11,3 | 3,1 a 10,7 |
| Árvores prevendo variação (PROFVAR_) | 1,6 a 3,6 | 1,2 a 2,9 |

A soja é mais difícil de bater que o milho na série diária. Na semanal, a combinação NAIVE + RET
ganha do ingênuo de forma significativa, como no milho.

## Os dois tipos de teste

Os dois testam o modelo em **dados do passado que já conhecemos**, escondendo do modelo o que vem
depois e comparando a previsão com o preço real.

**Holdout 80/20 (v4, `make holdout`)**: o teste clássico. A série é cortada em 80% / 20%:

- treino: ago/2001 a jan/2021 (1018 semanas);
- teste: fev/2021 a dez/2025 (255 semanas).

Os modelos são treinados **uma única vez**, só com o treino. Dois modos:

- `passo` (padrão): em cada semana do teste, o modelo prevê H semanas à frente usando o preço real
  até aquela semana, como seria no uso real (chega um preço novo por semana), mas sem retreinar.
- `cego`: uma única previsão feita em jan/2021 para os 5 anos seguintes, sem ver nenhum preço real
  do teste. Mostra até onde dá para prever "às cegas".

**Backtest (v3, `make backtest`)**: repete o teste em várias datas. A cada 4 semanas entre
jan/2023 e dez/2025 (39 datas), o modelo é **retreinado** com tudo que existia até ali e prevê
1, 4, 12 e 26 semanas à frente. Mede melhor como o modelo se comportaria sendo usado de verdade.

### Janela de treino: expansiva × deslizante

- **Expansiva (padrão):** o modelo estuda todo o passado, de 2001 até a data da previsão. A cada
  previsão nova a janela cresce.
- **Deslizante (`JANELA=N`):** o modelo estuda só os últimos N anos antes da previsão. A janela
  "anda" junto com o tempo, sempre do mesmo tamanho. A ideia é esquecer um passado muito antigo
  (preços de R$ 10 em 2001) que talvez não represente mais o mercado de hoje.
- O Chronos não é treinado, então continua recebendo o histórico inteiro nas duas opções.
- No código: `inicio_janela()` em `v3_backtest/backtest.py` e a opção `--janela-anos` nos dois scripts.

Primeiro resultado (holdout, 1 semana, janela de 10 anos = treino de 2011 a 2021): quase nada
muda. RET_RIDGE e RET_EXTRATREES vão de 0,90 para 0,89 (continuam significativamente melhores
que o ingênuo); PROF_RIDGE piora de 1,00 para 1,20; as árvores do professor continuam ruins
(5,7 a 10,6), porque o preço máximo dentro da janela continua sendo R$ 74.

### As três "janelas" de tempo (todas configuráveis)

Exemplo: "treinar com os últimos 10 anos, olhando as últimas 4 semanas, para prever 2 semanas à frente"
= `make holdout JANELA=10 ENTRADA=4 H=2`.

| Opção | Pergunta que responde | Unidade | Padrão |
|---|---|---|---|
| `H` | Quão longe o modelo prevê? | semanas | 1 (holdout) / 1, 4, 12, 26 (backtest) |
| `ENTRADA` | Quantas semanas passadas o modelo **olha** para fazer cada previsão? | semanas | cada modelo usa até 52 semanas |
| `JANELA` | Com quantos anos de dados o modelo **estuda** (é treinado)? | anos | todo o histórico desde 2001 |

O que `ENTRADA=N` muda em cada família:

- **Professor (PROF_, PROFVAR_):** ficam só as variáveis do script dele que cabem em N semanas
  (`lag_1` a `lag_N` e as médias/mín/máx móveis de até N semanas). Variáveis de calendário (`t`,
  `ano`, época do ano) e os preços das praças na semana atual continuam. Função
  `colunas_prof_entrada()` em `v3_backtest/backtest.py`.
- **Retorno (RET_):** as variações semana a semana dentro da janela (N preços = N-1 variações) e as
  médias/volatilidades que cabem nela. Função `features_diretas()`.
- **Chronos:** recebe só os últimos N preços como histórico. Função `contexto()`.
- **NAIVE** já usa só a última semana; **SNAIVE** (mesma semana do ano anterior) continua igual.

Primeiro teste (holdout, 2 semanas à frente, só alguns modelos), erro relativo ao ingênuo:

| Modelo | ENTRADA=1 | ENTRADA=4 | ENTRADA=52 |
|---|---|---|---|
| RET_RIDGE | 0,99 | **0,92** | 0,96 |
| COMBO_NAIVE_RET | 0,99 | **0,92** | 0,92 |
| PROF_RIDGE | 1,00 | **0,96** | 1,10 |
| CHRONOS_BOLT_SMALL | 1,00 | 0,98 | 1,02 |

Olhar só o último mês funcionou tão bem quanto ou melhor que olhar um ano inteiro; olhar só 1 semana
não basta. Ainda falta rodar com todos os modelos e outros horizontes para confirmar.

## Os modelos

| Família | Modelos | Como prevê |
|---|---|---|
| Referência | `NAIVE`, `SNAIVE` | Repete o último preço / o preço da mesma semana do ano anterior |
| Professor | `PROF_RIDGE`, `PROF_ELASTICNET`, `PROF_RF`, `PROF_EXTRATREES`, `PROF_GBR`, `PROF_HISTGBR`, `PROF_XGBOOST`, `PROF_LIGHTGBM` | Script do professor: prevê o **preço**, uma semana por vez (recursivo) |
| Professor prevendo variação | `PROFVAR_RIDGE`, ..., `PROFVAR_LIGHTGBM` (os mesmos 8) | Mesmos modelos, variáveis e parâmetros do professor, mas aprendendo a **variação** semanal do preço em vez do preço |
| Retorno | `RET_RIDGE`, `RET_EXTRATREES`, `RET_GBR` | Prevê a **variação** do preço até a data alvo, um modelo por horizonte, com variáveis também em variação |
| Chronos | `CHRONOS_BOLT_SMALL`, `CHRONOS_2` | Modelos pré-treinados da Amazon, usados sem treino; dão também uma faixa de incerteza |
| Combinações | `COMBO_NAIVE_RET` (média de NAIVE e RET_RIDGE), `COMBO_RET_CHRONOS` (RET_RIDGE e CHRONOS_2), `COMBO_LINEARES` (PROF_RIDGE, PROF_ELASTICNET e RET_RIDGE), `COMBO_MEDIANA` (mediana de todos) | Média das previsões dos outros modelos; calculadas automaticamente. As combinações foram fixadas antes de olhar o teste: escolher as melhores pelo resultado do teste seria trapaça |

## Como ler os resultados

Arquivos em `v4_holdout/outputs/<modo>/` (holdout) e `v3_backtest/outputs/` (backtest):

- `metricas.csv` / `metricas_por_horizonte.csv`: uma linha por modelo.
  - `MAE`: erro médio em R$/saca.
  - `MSE`: média dos erros **ao quadrado** (em R$²). Erros grandes pesam muito mais: errar R$ 10
    uma vez pesa como errar R$ 1 cem vezes.
  - `RMSE`: raiz do MSE, de volta em R$/saca. Fica sempre ≥ MAE; quanto maior a distância entre os
    dois, mais o modelo tem erros grandes de vez em quando.
  - `RMSE_rel_naive`: RMSE do modelo ÷ RMSE do `NAIVE` (< 1 = melhor). Comparar com o
    `MAE_rel_naive` mostra se o modelo ganha nos erros do dia a dia (MAE) ou nos erros grandes (RMSE).
  - `MAPE_percent`: erro médio em % do preço.
  - `MAE_rel_naive`: MAE do modelo ÷ MAE do `NAIVE`. **Métrica principal: abaixo de 1 o modelo
    ganha de repetir o último preço.**
  - `acerto_direcao_percent`: % de acerto de "o preço sobe ou cai" (50% = moeda).
  - `cobertura_80_percent` (Chronos): % de vezes em que o preço real caiu na faixa de 80%.
  - `dm_pvalor` e `vs_naive`: teste de Diebold-Mariano, que diz se a diferença para o `NAIVE` é
    estatisticamente significativa ou pode ser sorte. `p < 0,05` = significativa ("melhor" ou
    "pior"); acima disso, "sem diferenca". Nos gráficos de ranking, os significativos têm `*`.
- `previsoes.csv`: cada previsão feita, com a data, o valor previsto e o real.
- Gráficos do holdout:
  - `01_visao_geral.png`: série inteira com o treino e o teste, e o melhor modelo de cada família.
  - `02_todos_os_modelos.png`: um painel por modelo, real × previsto no teste.
  - `03_erro_relativo.png`: ranking do erro relativo ao `NAIVE`.
  - `04_erro_ao_longo_do_tempo.png`: em que períodos cada modelo erra mais.
  - `05_professor_preco_vs_variacao.png`: cada modelo do professor prevendo o preço × a variação.

## Resultados

### Holdout 80/20 (treino até jan/2021, teste fev/2021–dez/2025)

Erro relativo ao `NAIVE` (< 1 = melhor):

| Modelo | 1 semana | 4 semanas | 12 semanas |
|---|---|---|---|
| RET_RIDGE | **0,90** | **0,99** | 1,11 |
| RET_EXTRATREES | **0,90** | 1,02 | 1,12 |
| PROF_ELASTICNET | 0,97 | 1,06 | **0,99** |
| PROF_RIDGE | 1,00 | 1,07 | 1,01 |
| CHRONOS_BOLT_SMALL | 1,12 | 1,06 | 1,04 |
| CHRONOS_2 | 1,15 | 1,06 | 1,14 |
| PROF_RF / EXTRATREES / GBR | 6,4–6,9 | 2,7–2,9 | 1,5–1,6 |
| PROF_XGBOOST / LIGHTGBM / HISTGBR | 9,1–10,7 | 3,4–3,6 | 1,9–2,0 |
| PROFVAR (árvores) | 1,1–2,1 | 1,7–2,8 | 2,7–4,5 |
| PROFVAR_ELASTICNET | 1,00 | 1,01 | 1,04 |
| COMBO_NAIVE_RET | **0,89** | **0,95** | 1,00 |
| COMBO_LINEARES | 0,92 | 0,98 | **0,98** |

- **Em 1 semana, os modelos de retorno e a combinação NAIVE + RET_RIDGE ganham do ingênuo por
  10–11%, e a diferença é estatisticamente significativa** (Diebold-Mariano, p < 0,05). É o único
  caso, nos dois testes, com uma vantagem comprovada.
- **Os modelos de árvore do professor falham feio no holdout.** O preço máximo no treino é
  R$ 74, mas no teste chega a R$ 97 (2021–22). Árvores não conseguem prever um valor acima do
  que viram no treino e ficam "presas" em ~R$ 71 (ver `02_todos_os_modelos.png`). Os modelos de
  retorno não têm esse problema porque preveem a variação, não o preço.
- **Trocar o alvo do professor para variação (PROFVAR) resolve o problema em 1 semana**: as
  árvores caem de 6–11× para 1,1–2,1× o erro do ingênuo. **Mas piora em 4 e 12 semanas**: a
  previsão recursiva soma 4 ou 12 variações previstas em sequência, e o erro se acumula. As
  variáveis do professor continuam em nível de preço (lags, médias, `t`, `ano`); no topo do treino
  (fim de 2020) o preço só subia, então os modelos tendem a prever alta contínua quando veem preço
  alto. O PROFVAR_RIDGE piora até em 1 semana pelo mesmo motivo. Para prever variação, as variáveis
  também precisam estar em variação, que é o que os modelos RET_ fazem.
- Em 4 e 12 semanas, nenhum modelo se afasta de forma significativa do ingênuo.

### Backtest (39 datas, jan/2023–dez/2025)

| Horizonte | Melhor modelo | Erro rel. | MAPE do ingênuo |
|---|---|---|---|
| 1 semana | RET_RIDGE | 1,04 | 1,5% |
| 4 semanas | CHRONOS_2 | 1,00 | 5,0% |
| 12 semanas | PROF_ELASTICNET | 0,97 | 11,9% |
| 26 semanas | CHRONOS_2 | 0,96 | 19,3% |

Ranking completo em `v3_backtest/outputs/ranking_mae_relativo.csv`. Aqui os modelos de árvore do
professor vão bem melhor que no holdout porque são retreinados a cada data e já viram preços altos.

A combinação `COMBO_NAIVE_RET` fica abaixo de 1 em todos os horizontes (0,96 a 0,97), a mais
estável de todas, mas com só 39 datas a diferença não é significativa. Só PROF_RIDGE em 26 semanas
(0,98, p = 0,03) passa no teste de significância. Os modelos PROFVAR ainda não rodaram no backtest
(`make backtest MODELOS=PROFVAR`, ~7 min).

### Conclusões até agora

- Só com o histórico do preço, quase nenhum modelo supera de forma clara o "repetir o último preço".
  A exceção comprovada é 1 semana à frente no holdout (RET_ e COMBO_NAIVE_RET, ~10% melhor).
- Prever a variação em vez do preço é mais robusto quando o preço muda de patamar, desde que as
  variáveis também estejam em variação (RET_). Só trocar o alvo (PROFVAR) ajuda em 1 semana e
  atrapalha em horizontes maiores.
- Combinar o ingênuo com um modelo (COMBO_NAIVE_RET) é a opção mais estável nos dois testes.
- RET_RIDGE acerta a direção (sobe/cai) em 62–71% dos casos, nos dois testes.
- A faixa de 80% do Chronos-2 contém o valor real em 78–88% dos casos (bem calibrada).
- O script do professor, rodado direto na base, treinava só com ~95 semanas porque a praça Pitanga
  só começa em 2023. A v3 e a v4 removem as praças com mais de 10% de lacunas.
- Próximo passo: variáveis externas (milho em Chicago, câmbio PTAX, CEPEA, soja).

## Estrutura

```
Makefile                comandos (make ajuda)
requirements.txt
dados_milho/
  brutos/sima/          planilhas originais do SIMA, uma pasta por ano
  extrair_milho.py      lê as planilhas e gera os CSVs abaixo
  milho_diario.csv      preço diário por praça
  milho_semanal.csv     média semanal (semana terminando na sexta)
app/                    interface web (FastAPI + banco); ver "Interface no navegador"
Dockerfile, docker-compose.yml
dados_soja/
  cepea-*.xls           planilha baixada do CEPEA (indicador da soja, Paraná)
  extrair_soja.py       lê a planilha e gera soja_diario.csv e soja_semanal.csv
v1_janelas_ml/          1ª versão: KNN/RF/Ridge sobre janelas de 200 semanas (sem avaliação)
v2_modelo_professor/    script do professor (teste de 1 passo nas últimas 52 semanas)
v3_backtest/            backtest em várias datas, vários horizontes
v4_holdout/             holdout 80/20 (usa as funções do v3)
v5_diario_semanal/      holdout 80/20 diário ou semanal + previsão futura (usa v3 e v4)
  previsao.py           o holdout (make v5)
  comparar.py           compara as frequências (make v5-comparar)
notebooks/              exploração
```

Observações técnicas:

- No Mac, a `libomp` do XGBoost/LightGBM conflita com a do PyTorch (Chronos) e trava o Python.
  Por isso o Chronos roda num processo separado; não é preciso fazer nada.
- A v1 depende de `ReduzirGranilaridade.py` (gera os arquivos `_GRAN.csv`), que não está no
  projeto; veja `v1_janelas_ml/ToRun.txt`.
- Sem `make`, os comandos equivalentes estão no próprio `Makefile`, por exemplo
  `.venv/bin/python v4_holdout/holdout.py --horizonte 4 --models PROF`.

# TCC – Previsão do preço do milho (Paraná, SIMA/DERAL)

Preço semanal do milho (R$/saca 60 kg) nas praças do Paraná, de 2001 a 2025.
Alvo: `media_pr` (média simples das praças).

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
| `make dados` | Regera `dados/milho_semanal.csv` a partir das planilhas do SIMA | ~1 min |
| `make professor` | Roda o script original do professor (v2) | ~10 s |
| `make holdout` | Holdout 80/20, previsão 1 semana à frente (v4) | ~20 s |
| `make holdout H=4` | O mesmo, 4 semanas à frente (vale qualquer valor de 1 a 52) | ~2 min (H=12: ~5 min) |
| `make holdout-cego` | Holdout 80/20 com uma única previsão para os 20% inteiros | ~1 min |
| `make holdout-todos` | Holdout em 1, 4 e 12 semanas + cego | ~8 min |
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
dados/
  brutos/sima/          planilhas originais do SIMA, uma pasta por ano
  extrair_milho.py      lê as planilhas e gera os CSVs abaixo
  milho_diario.csv      preço diário por praça
  milho_semanal.csv     média semanal (semana terminando na sexta)
v1_janelas_ml/          1ª versão: KNN/RF/Ridge sobre janelas de 200 semanas (sem avaliação)
v2_modelo_professor/    script do professor (teste de 1 passo nas últimas 52 semanas)
v3_backtest/            backtest em várias datas, vários horizontes
v4_holdout/             holdout 80/20 (usa as funções do v3)
notebooks/              exploração
```

Observações técnicas:

- No Mac, a `libomp` do XGBoost/LightGBM conflita com a do PyTorch (Chronos) e trava o Python.
  Por isso o Chronos roda num processo separado; não é preciso fazer nada.
- A v1 depende de `ReduzirGranilaridade.py` (gera os arquivos `_GRAN.csv`), que não está no
  projeto; veja `v1_janelas_ml/ToRun.txt`.
- Sem `make`, os comandos equivalentes estão no próprio `Makefile`, por exemplo
  `.venv/bin/python v4_holdout/holdout.py --horizonte 4 --models PROF`.

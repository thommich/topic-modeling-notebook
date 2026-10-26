# Diagnóstico de cohorts a partir de um cubo Pandas

Use Python 3.10+ com as dependências de `requirements.txt`.

`analisar_cubo` recebe **um DataFrame já agregado em formato de cubo**. Não lê
registros individuais nem constrói o cubo. Cada linha representa uma célula
observada ou uma totalização do cubo.

## Contrato de entrada

As primeiras colunas são dimensões de segmentação. Opcionalmente, uma delas
pode ser `safra` e outra `amostra`. **As últimas três colunas**, nesta ordem,
devem conter:

1. Quantidade de maus (`target = 1`).
2. Quantidade com performance disponível (`target = 0` ou `1`).
3. Quantidade total, incluindo target ausente.

O cubo deve incluir o total geral e os rollups de todas as combinações de
características analisadas. Se usar `safra` ou `amostra`, inclua também os
rollups correspondentes. Em cada célula deve valer
`0 <= maus <= performance <= total`. As contagens das células de uma
partição devem reproduzir as três contagens do total geral.

Por padrão, `NaN`/`None` identifica uma dimensão totalizada. Caso haja
categorias genuinamente nulas, use **outro marcador exclusivo** nas células
totalizadoras, por exemplo `"<TOTAL>"`, e passe `marcador_total="<TOTAL>"`.
Sem essa distinção, um nulo real e um subtotal SQL são indistinguíveis.

## Uso

```python
from analise_cohorts import CriteriosElegibilidade, analisar_cubo

# df_cubo já é um pandas.DataFrame; as últimas colunas são
# ["qtd_maus", "qtd_performance", "qtd_total"].
resultado = analisar_cubo(
    df_cubo,
    col_safra="safra",        # omita se não houver dimensão temporal
    col_amostra="amostra",    # omita se não houver divisão de amostras
    marcador_total="<TOTAL>",
    criterios=CriteriosElegibilidade(
        minimo_total=1000,
        minimo_performance=500,
        minimo_bons=100,
        minimo_maus=30,
        cobertura_minima=0.80,
    ),
)

ranking = resultado.combination_summary
grupos = resultado.cohort_detail
evolucao = resultado.cohort_temporal
incrementos = resultado.combination_comparison

print(ranking[["ranking", "combinacao", "qtd_grupos", "cobertura_elegivel",
               "v_cramer", "ganho_v_melhor_subset", "fronteira_pareto"]].head(20))
```

Se todas as primeiras colunas forem características, `caracteristicas` é
inferido automaticamente. Também é possível fornecê-lo explicitamente para
definir a ordem dos bits do `combination_id`. Esse identificador é a chave de
ligação entre as tabelas; o rótulo textual é apenas para leitura.

## Saídas e critérios

- `combination_summary`: uma linha por subconjunto não vazio de
  características (255 linhas para oito características). Mostra grupos
  observados, suficiência amostral, cobertura, dispersão do bad rate, V de
  Cramér, IV, ganho sobre o melhor subconjunto imediato, estabilidade e
  fronteira de Pareto. O ranking põe primeiro as combinações viáveis e depois
  ordena por V, cobertura e complexidade. `viavel_global` exige ao menos dois
  grupos elegíveis e a cobertura mínima configurada no total geral; quando há
  `col_amostra`, `viavel_amostras` exige isso em **cada** amostra e `viavel`
  exige ambos. Sem `col_amostra`, `viavel` equivale a `viavel_global`.
- `cohort_detail`: uma linha por grupo observado em cada combinação, com
  volumes, bons, maus, performance ausente, participação no público total,
  bad rate, diferença para a bad rate geral em pontos percentuais, intervalo
  de Wilson e elegibilidade. Dimensões fora da combinação mantêm o marcador
  de totalização.
- `cohort_temporal`: grupo × safra, se `col_safra` for informado. A
  participação usa o total da safra e a bad rate relativa usa a bad rate da
  própria safra. A estabilidade da participação considera zero quando um
  grupo desaparece em determinada safra.
- `combination_comparison`: uma linha para cada relação entre uma combinação
  e um de seus subconjuntos imediatos, com variação de V, cobertura e número
  de grupos.

Todas as bad rates, participações e coberturas são frações entre 0 e 1; campos
com sufixo `_pp` estão em **pontos percentuais**. Bad rate e medidas de risco
usam `qtd_performance` como denominador. As medidas de risco da combinação
consideram somente grupos elegíveis, ponderados pela quantidade com
performance disponível. O IV usa suavização de 0,5 por grupo para evitar
logaritmos indefinidos quando há zero bons ou maus. A fronteira de Pareto
maximiza V, cobertura e, se houver safra, persistência do sinal de risco, e
minimiza a quantidade de grupos, entre as combinações viáveis.

Se `col_amostra` for informado, o resumo também traz o menor e maior V e a
menor cobertura elegível entre as amostras. Essas medidas usam os mesmos
critérios mínimos em cada amostra e ajudam a detectar dependência de uma
partição específica. Elas não substituem uma validação temporal final.

## Segunda fase: qualidade e poder preditivo das features

`avaliacao_features.py` recebe uma base por registro com as mesmas dimensões
do cubo, `safra`, `amostra`, target 0/1/NaN e colunas de features. Antes de
avaliá-las, confere as contagens de cada cohort contra a primeira fase. A
função `avaliar_features` aceita features numéricas e categóricas e, por
padrão, seleciona as três primeiras combinações viáveis de duas ou três
características do ranking. Também é possível passar os `combination_id`
explicitamente.

```python
from avaliacao_features import CriteriosFeature, avaliar_features

piloto = avaliar_features(
    dados_features,       # uma linha por registro, inclusive target ausente
    resultado,            # saída de analisar_cubo(df_cubo, ...)
    features=["bureau_score", "interno_score", "sinal_canal"],
    combinacoes=[7, 35], # opcional; IDs do combination_summary
    criterios=CriteriosFeature(
        minimo_avaliavel=500, minimo_bons=300, minimo_maus=70,
        cobertura_minima=0.65, auc_ci_inferior_min=0.55,
        iv_minimo_categorica=0.02,
    ),
)
```

As quatro tabelas dessa fase são:

- `feature_detail`: feature × cohort × amostra, com cobertura, missing,
  bons/maus avaliáveis, AUC, intervalo aproximado da AUC, IV, limite inferior
  do IV e marcação de aprovação.
- `feature_temporal`: as mesmas métricas por safra.
- `feature_summary`: comparação de cada feature entre cohorts na validação,
  incluindo menor cobertura, menor limite inferior de AUC/IV, dispersão
  temporal, diferença média validação menos desenvolvimento e quantos grupos
  atendem aos critérios.
- `combination_feature_summary`: número de features aprovadas em **todos** os
  cohorts, variação média de cobertura e AUC entre grupos e concordância
  média de Spearman dos rankings de IV entre cohorts no desenvolvimento.

Para features numéricas, a direção da AUC e os cortes de quantis usados no
IV são fixados na amostra de desenvolvimento e reaplicados na validação. A
aprovação exige cobertura e eventos suficientes, além de limite inferior do
intervalo aproximado da AUC acima do mínimo configurado. Para categóricas, a
aprovação usa o limite inferior da distribuição posterior do IV, obtido por
200 sorteios Dirichlet com suavização de 0,5. Missing é um bin explícito no
IV. AUC de feature constante ou de uma classe de target ausente fica nula.

## Exemplo simulado das duas fases

`exemplo_simulado.py` gera um cubo reproduzível com oito características (duas
com três categorias), três safras e duas amostras. A dimensão `garantia` tem
uma categoria realmente nula, distinta de `<TOTAL>`. O script também gera
uma base **por registro** com target 0/1/NaN e nove features: scores com
cobertura dependente de origem ou relacionamento, uma feature categórica,
uma feature cujo poder cai na validação, ruído e uma constante. As contagens
da base individual conciliam exatamente com as do cubo. O script grava e relê
ambos os CSVs antes de executar as duas fases:

```bash
python exemplo_simulado.py
```

Os dez CSVs ficam em `exemplo_simulado/`: o cubo, a base de features e as
quatro tabelas de cada fase. Para gerar outra realização ou escolher a pasta
de saída, use `--seed` e `--saida`. Os arquivos presentes neste projeto foram
gerados com a semente padrão `20261026`. O total geral tem 309.362 registros,
dos quais 262.974 com performance e 56.356 maus. A primeira fase produz 255
combinações, das quais 231 são viáveis no total e em cada amostra. O piloto
avalia três combinações, cada uma com 18 cohorts, e nove features. Esses
números são **inteiramente sintéticos**; a relação gerada entre target e
features serve para testar o método, não para recomendar cohorts reais.

## Execução dos testes

Com Pandas e NumPy instalados:

```bash
python -m unittest -v test_analise_cohorts.py test_avaliacao_features.py
```

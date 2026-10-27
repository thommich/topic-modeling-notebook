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






A abordagem que você está construindo faz bastante sentido para o desenvolvimento do modelo de crediário. Eu, porém, faria uma distinção importante: identificar combinações de características que produzem grandes diferenças de bad rate não é necessariamente o mesmo que identificar as melhores combinações para definir cohorts de feature selection.

O primeiro problema é de heterogeneidade de risco; o segundo é de heterogeneidade do comportamento das features. Os dois estão relacionados, mas não são equivalentes.

Minha proposta seria estruturar a análise em duas etapas:

* Diagnóstico das segmentações: avaliar todas as combinações das oito características, quantificando diferenças de risco, representatividade, granularidade e estabilidade.

* Seleção dos cohorts: escolher as segmentações que melhor ajudam a identificar features robustas, considerando também a viabilidade estatística das análises por subgrupo.

O principal produto da primeira etapa seria um ranking comparativo de combinações, acompanhado por uma tabela detalhada dos grupos produzidos por cada combinação.

### 1\. Organizando o universo de combinações

Assumindo que suas oito características sejam binárias (flags 0/1), existem 255 combinações não vazias de características.

| Características combinadas | Nº de combinações | Máximo de grupos por combinação |
| -------------------------- | ----------------- | ------------------------------- |
| 1                          | 8                 | 2                               |
| 2                          | 28                | 4                               |
| 3                          | 56                | 8                               |
| 4                          | 70                | 16                              |
| 5                          | 56                | 32                              |
| 6                          | 28                | 64                              |
| 7                          | 8                 | 128                             |
| 8                          | 1                 | 256                             |
| Total                      | 255               | 6.560 grupos potenciais         |

Os 6.560 grupos correspondem à soma das possíveis células em todas as 255 partições, não a 6.560 públicos distintos e mutuamente exclusivos.

Por exemplo, considere três características:

* A: cliente possui determinado relacionamento com a instituição.

* B: cliente pertence a determinada faixa de exposição.

* C: contrato possui determinada característica de origem.

A combinação `A+B` produz quatro grupos: `00`, `01`, `10` e `11`.

Já `A+B+C` produz oito grupos. Cada uma dessas combinações representa uma maneira diferente de particionar a mesma população.

Essa distinção determina como eu construiria o relatório: a unidade principal de comparação deve ser a combinação de características, e não cada grupo isoladamente.

Assim, você teria 255 linhas em um relatório sintético e uma segunda tabela contendo até 6.560 linhas para permitir a exploração detalhada de cada combinação.

### 2\. Quais métricas eu calcularia para cada combinação?

Eu dividiria os indicadores em quatro famílias. Essa separação facilita bastante a interpretação dos resultados, porque uma combinação pode ser excelente em discriminação de risco e ruim em representatividade, por exemplo.

### A. Representatividade e suficiência amostral

São os indicadores que permitem avaliar se os cohorts gerados terão tamanho suficiente para as etapas seguintes de feature selection.

| Indicador                | O que representa                                                        |
| ------------------------ | ----------------------------------------------------------------------- |
| Quantidade de grupos     | Número de cohorts efetivamente observados                               |
| Menor grupo (%)          | Participação do menor grupo na população                                |
| Menor grupo (N)          | Quantidade de registros do menor grupo                                  |
| Menor quantidade de bads | Número de eventos no grupo com menos defaults                           |
| Grupos elegíveis (%)     | Percentual de grupos que cumprem os critérios mínimos                   |
| Cobertura elegível (%)   | Percentual da população presente em grupos estatisticamente utilizáveis |

Eu daria bastante atenção à última métrica.

Imagine duas segmentações:

* Segmentação A: produz quatro grupos com grandes diferenças de bad rate, mas dois deles representam apenas 0,2% da população.

* Segmentação B: produz quatro grupos com diferenças de bad rate ligeiramente menores, mas todos têm representatividade relevante.

Para seu objetivo, a segmentação B pode ser consideravelmente mais útil.

Além disso, eu definiria os limites mínimos considerando tanto a quantidade de registros quanto a quantidade de bons e maus. Um cohort grande, mas com pouquíssimos eventos de default, pode ser inadequado para avaliar poder preditivo.

### B. Heterogeneidade de risco

Aqui está o núcleo da análise que você já começou.

| Indicador                     | Interpretação                                             |
| ----------------------------- | --------------------------------------------------------- |
| BR mínimo                     | Menor bad rate entre os grupos elegíveis                  |
| BR máximo                     | Maior bad rate entre os grupos elegíveis                  |
| Amplitude do BR               | BR máximo menos BR mínimo, em pontos percentuais          |
| Razão de risco                | BR máximo dividido pelo BR mínimo                         |
| Desvio-padrão ponderado do BR | Dispersão dos bad rates considerando o tamanho dos grupos |
| V de Cramér                   | Força da associação entre a segmentação e o default       |
| IV entre cohorts              | Separação entre a distribuição de bons e maus             |

Eu não escolheria a amplitude de BR como métrica principal.

Ela é facilmente influenciada por um grupo pequeno com bad rate extremo. Para complementar essa análise, utilizaria uma medida de dispersão ponderada.

Se é a bad rate do grupo , é seu tamanho e a bad rate da população:

D_{BR}=\\sqrt{\\sum_g\\frac{n_g}{N}(p_g-p)^2}

Essa métrica expressa a heterogeneidade do risco na escala da própria bad rate, em vez de considerar apenas os grupos extremos.

Uma segunda medida que considero especialmente interessante é o V de Cramér. Como a variável resposta é binária, ele pode ser calculado diretamente:

V=\\sqrt{\\frac{\\sum_g (n_g/N)(p_g-p)^2}{p(1-p)}}

O indicador permite comparar a associação das diferentes segmentações com a performance, usando uma escala normalizada entre 0 e 1.

O cuidado é que segmentações com mais características e, consequentemente, mais grupos, tendem a apresentar indicadores de associação maiores na própria amostra, mesmo quando parte do ganho é apenas ruído.

Por isso, eu utilizaria o V de Cramér como indicador de ordenação inicial, mas avaliaria a robustez fora da amostra e o ganho incremental trazido por cada nova característica.

### C. Estabilidade temporal

Este é um componente que eu acrescentaria desde o início, ainda que sua análise atual esteja focada em diferenças de bad rate.

Uma boa segmentação precisa produzir diferenças de performance suficientemente persistentes.

Eu avaliaria:

* Estabilidade da participação populacional de cada grupo ao longo das safras.

* Estabilidade do bad rate relativo à média da respectiva safra.

* Persistência da ordenação de risco entre os grupos.

* Estabilidade dos indicadores de separação de risco ao longo do tempo.

Por exemplo, um cohort cuja bad rate permanece sistematicamente acima da média da safra é mais interessante do que um cohort que alterna frequentemente entre os extremos.

Usar a diferença ou a razão em relação à bad rate geral de cada safra ajuda a separar mudanças gerais no nível de risco de mudanças específicas do cohort.

### D. Complexidade e ganho incremental

Eu incluiria indicadores que respondam a uma pergunta:

Ao acrescentar mais uma característica à segmentação, o que efetivamente ganho?

Imagine que as características A e B já produzem cohorts com diferenças expressivas de risco. Acrescentar C pode duplicar a quantidade de grupos, mas acrescentar pouquíssima informação.

Para cada combinação, calcularia o ganho incremental de separação em relação à melhor combinação imediatamente mais simples contida nela.

Exemplo:

| Combinação | V de Cramér | Ganho ilustrativo |
| ---------- | ----------- | ----------------- |
| A          | 0,12        | —                 |
| B          | 0,10        | —                 |
| A+B        | 0,18        | +0,06             |
| A+B+C      | 0,19        | +0,01             |
| A+B+C+D    | 0,19        | ≈0,00             |

Valores inteiramente fictícios, apenas para ilustrar a análise incremental.

Eu priorizaria esse ganho na validação, idealmente com uma métrica fora da amostra ou corrigida pela complexidade. Não trataria pequenas diferenças de V calculadas na mesma base como ganhos comprovados.

Nesse exemplo, `A+B` seria uma candidata bastante forte: entrega boa separação de risco sem exigir muitos grupos.

### 3\. Como eu organizaria o relatório

Minha recomendação é construir três camadas, cada uma com um nível diferente de granularidade.

### Camada 1 — Ranking executivo das 255 combinações

Visão sintética das combinações

Exemplo ilustrativo

| Combinação | Grupos | Cobertura válida | Δ BR    | V    | Estabilidade |
| ---------- | ------ | ---------------- | ------- | ---- | ------------ |
| A+B        | 4      | 99,1%            | 8,2 pp  | 0,18 | Alta         |
| A+C+D      | 8      | 96,5%            | 12,4 pp | 0,21 | Alta         |
| B+E+F      | 8      | 76,2%            | 16,8 pp | 0,25 | Baixa        |
| A+B+C+D    | 16     | 88,3%            | 13,1 pp | 0,22 | Média        |

As 255 combinações poderiam ser ordenadas por poder de separação, suficiência amostral, estabilidade ou um score de priorização.

A tabela completa conteria também o número de características, os grupos elegíveis, o menor número de bads, o desvio-padrão ponderado do BR, o IV, o ganho incremental e os indicadores de estabilidade.

Ela seria a base para o ranking e a comparação entre candidatos.

### Camada 2 — Detalhamento dos grupos de cada combinação

Aqui o objetivo é entender por que determinada combinação recebeu uma boa ou má classificação.

Considere a combinação `A+B`:

| A   | B   | % público | Bons   | Bads  | Bad rate |
| --- | --- | --------- | ------ | ----- | -------- |
| 0   | 0   | 40%       | 36.000 | 4.000 | 10%      |
| 0   | 1   | 30%       | 25.500 | 4.500 | 15%      |
| 1   | 0   | 20%       | 16.000 | 4.000 | 20%      |
| 1   | 1   | 10%       | 7.000  | 3.000 | 30%      |

Exemplo com 100 mil registros e números fictícios.

Além dessas métricas, incluiria a bad rate relativa ao público total, os intervalos de confiança, a representatividade por safra e uma marcação de elegibilidade estatística.

Essa tabela detalhada seria vinculada à primeira pelo `combination_id`, permitindo consultar a composição dos grupos de qualquer uma das 255 combinações.

### Camada 3 — Robustez das combinações finalistas

Depois de selecionar, por exemplo, as 10–20 combinações mais promissoras, eu faria uma análise temporal mais profunda.

Exemplo de visualização — bad rate relativa por safra

Valores fictícios. Diferença entre a BR do cohort e a BR geral da safra, em pontos percentuais.

Exemplo ilustrativo

Eu avaliaria tanto a permanência da diferença de risco quanto a estabilidade do tamanho dos grupos. Uma combinação pode manter boa separação de risco, mas perder representatividade ao longo do tempo, o que também é relevante para a utilização futura no pipeline.

### 4\. Como escolher as combinações finalistas

Eu não produziria um único ranking ordenado pela amplitude de bad rate. Utilizaria um processo de seleção em três estágios.

Etapa 1 — Filtros de viabilidade

Descartar ou sinalizar combinações com muitos grupos sem volume suficiente, baixa cobertura amostral ou características indisponíveis no momento relevante de observação. Os limites de tamanho e eventos devem ser calibrados ao volume do seu público e às análises posteriores.

Etapa 2 — Discriminação e estabilidade

Priorizar combinações com heterogeneidade de BR expressiva, associação robusta com default e diferenças persistentes entre safras. Comparar o desempenho em validações temporais.

Etapa 3 — Parcimônia e ganho incremental

Comparar cada combinação com suas versões mais simples. Preferir a segmentação que forneça informação adicional relevante com o menor número possível de grupos, sem deixar desprotegidos subpúblicos estratégicos.

Uma alternativa interessante a criar imediatamente um score consolidado é identificar uma fronteira de Pareto, na qual cada combinação é avaliada por quatro dimensões: separação de risco, representatividade, estabilidade e complexidade.

Isso ajuda a evitar pesos arbitrários e permite enxergar segmentações que representam bons compromissos entre os objetivos.

Eu também compararia as combinações dentro de cada nível de complexidade. Por exemplo, primeiro identificaria as melhores entre as 28 combinações de duas características e as 56 de três características, para depois confrontá-las com as segmentações de quatro ou mais características.

### 5\. Como gerar o relatório de maneira eficiente

Considerando apenas oito flags binárias, há uma otimização interessante: você não precisa executar 255 consultas sobre a base individualizada.

Minha preferência seria:

* Pré-agregar a população pelas oito flags, safra, amostra e target, calculando quantidade de bons e maus.

* A partir dessa tabela agregada, reconstruir as 255 partições.

* Calcular as métricas de cada partição e consolidá-las nas tabelas de resumo e detalhamento.

A pré-agregação gera, no máximo, 256 perfis distintos de flags por safra e combinação de amostra/target. Todo o processamento combinatório seguinte pode ocorrer sobre uma tabela muito menor.

Em SQL, o operador `CUBE` permite produzir as agregações de todos os subconjuntos de colunas. O `GROUPING()` identifica se uma dimensão faz parte do agrupamento ou foi agregada, distinguindo o totalizador de um valor nulo real.

Um esqueleto em BigQuery SQL seria:

A partir do resultado, faria os cálculos estatísticos em Python (Pandas ou Polars), porque isso torna mais fácil produzir os rankings, filtros, gráficos e análises incrementais.

Para o relatório, minha estrutura final seria:

| Tabela                 | Granularidade                          | Finalidade                           |
| ---------------------- | -------------------------------------- | ------------------------------------ |
| combination_summary    | 1 linha por combinação                 | Ranking e indicadores consolidados   |
| cohort_detail          | 1 linha por grupo dentro da combinação | Volume, defaults, BR e elegibilidade |
| cohort_temporal        | Grupo × safra                          | Evolução temporal e estabilidade     |
| combination_comparison | Combinação × subconjunto anterior      | Ganho incremental e redundância      |

As quatro tabelas podem ser exportadas para Excel ou alimentar um dashboard em Power BI, Plotly ou Streamlit. O relatório executivo exibiria o ranking, enquanto os detalhes permaneceriam disponíveis para investigação.

### 6\. O passo adicional que eu considero mais importante

Voltando ao objetivo original: os cohorts serão utilizados para avaliar qualidade, poder preditivo e estabilidade das features que vieram de diferentes fontes via `LEFT JOIN`.

Eu acrescentaria uma segunda análise à metodologia, depois de identificar as segmentações com maiores diferenças de risco.

Verificaria se essas segmentações realmente produzem diferenças relevantes no comportamento das features.

É perfeitamente possível que dois grupos tenham bad rates muito distintas e, ainda assim, apresentem praticamente o mesmo ranking de variáveis, a mesma qualidade de dados e a mesma estabilidade.

Nesse caso, a segmentação diferencia risco, mas talvez não acrescente muito ao processo de feature selection.

Em contrapartida, dois grupos podem ter bad rates próximas, mas apresentar diferenças importantes de cobertura das fontes de dados ou do poder preditivo das variáveis.

Para seus objetivos, a segunda situação pode ser até mais relevante.

Por isso, nas combinações finalistas, eu faria um piloto com um subconjunto representativo das features, observando:

* Diferenças de cobertura, missing e outros indicadores de data quality entre cohorts.

* Diferenças de IV, AUC ou outra medida univariada pertinente entre cohorts.

* Estabilidade temporal do poder preditivo dentro dos grupos.

* Concordância dos rankings das melhores features entre os cohorts.

* Quantidade de features que passariam nos critérios mínimos em todos os grupos, considerando a incerteza das estimativas.

Isso permitiria avaliar diretamente a utilidade da segmentação para o pipeline, e não apenas sua capacidade de separar níveis de risco.

Também tomaria quatro cuidados específicos no contexto do modelo de crediário: garantir maturação equivalente dos targets (especialmente se utilizar 30M3, 60M6 e 90M12), avaliar possíveis diferenças estruturais entre as origens A e B, utilizar somente características observáveis na referência relevante e preservar um teste temporal final sem participação na escolha dos cohorts.

### 7\. Minha proposta para o relatório final

Eu organizaria a apresentação executiva em quatro páginas ou abas:

01

Visão geral

Distribuição das 255 combinações por quantidade de características, cobertura elegível e separação de risco.

02

Ranking de combinações

Tabela comparativa completa, ordenável e filtrável por todas as métricas calculadas.

03

Explorador de cohorts

Detalhamento dos grupos de uma combinação selecionada, incluindo volumes, bad rates e evolução temporal.

04

Shortlist e decisão

Comparação das finalistas, ganho incremental, robustez e justificativa dos cohorts recomendados.

Minha recomendação é usar a análise exploratória das 255 combinações para reduzir o espaço de busca, e não para decidir definitivamente os cohorts. Eu começaria selecionando três a cinco candidatas, preferencialmente com duas ou três características, e só então avaliaria a heterogeneidade do comportamento das features.

Se essas segmentações mais simples não cobrirem adequadamente algum público relevante ou apresentarem diferenças insuficientes, avançaria para combinações mais complexas ou para uma segmentação hierárquica, na qual os grupos raros sejam tratados explicitamente.

Isso aproxima a escolha dos cohorts da finalidade real do pipeline: selecionar um conjunto de features que seja preditivo, estável e utilizável de forma consistente nos diferentes perfis de clientes do modelo de crediário.

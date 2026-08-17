# Relatório Técnico — Tech Challenge Fase 1

**Projeto:** Classificação de tumores de mama (maligno vs. benigno) com Machine Learning
**Curso:** Pós-graduação em Inteligência Artificial — FIAP
**Repositório:** https://github.com/gustavofks/ML-breast-cancer

> Documento em construção. Cada seção é fechada ao final da fase correspondente do
> desenvolvimento; a revisão final e a exportação para PDF ocorrem na Fase 7.

---

## Sumário

1. [O problema](#1-o-problema)
2. [Análise exploratória dos dados](#2-análise-exploratória-dos-dados)
3. [Estratégias de pré-processamento](#3-estratégias-de-pré-processamento) *(Fase 3)*
4. [Modelos utilizados e justificativa](#4-modelos-utilizados-e-justificativa) *(Fase 4)*
5. [Resultados e interpretação](#5-resultados-e-interpretação) *(Fases 4 e 5)*
6. [Discussão crítica e uso na prática](#6-discussão-crítica-e-uso-na-prática) *(Fase 7)*

---

## 1. O problema

Uma rede de hospitais e centros de saúde especializados no atendimento à mulher busca implementar
um sistema inteligente de suporte ao diagnóstico, capaz de acelerar a triagem e apoiar decisões
clínicas diante de um volume crescente de pacientes.

Este projeto entrega a base de Machine Learning dessa solução: a classificação automática de
tumores de mama em **malignos** ou **benignos** a partir de características morfológicas de núcleos
celulares extraídas de imagens de punção aspirativa por agulha fina (PAAF).

**Base escolhida:** Breast Cancer Wisconsin (Diagnostic), disponível em
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data — 569 amostras e 30
características numéricas, com diagnóstico confirmado.

A base descreve 10 medidas do núcleo celular (raio, textura, perímetro, área, suavidade,
compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), cada uma em três
variantes: média da amostra (`_mean`), erro padrão da medida (`_se`) e média das três piores
células observadas (`_worst`).

**Premissa que orienta todo o projeto:** o modelo é ferramenta de triagem e segunda opinião. O
diagnóstico final é sempre responsabilidade do médico. Essa premissa define a métrica prioritária
(recall da classe maligna) e a exigência de explicabilidade das previsões.

---

## 2. Análise exploratória dos dados

Notebook correspondente: [`notebooks/01_exploratory_analysis.ipynb`](../notebooks/01_exploratory_analysis.ipynb)

### 2.1 Qualidade e integridade da base

A base tem 569 registros e **nenhum valor ausente real**. A única inconsistência encontrada é
estrutural: o cabeçalho do CSV termina em vírgula, o que faz o pandas criar uma coluna extra
`Unnamed: 32` inteiramente nula. Trata-se de artefato do arquivo, não de dado faltante — o
tratamento correto é descartar a coluna, e não imputar valores.

A coluna `id` também foi removida: é identificador do paciente, não tem relação causal com o
diagnóstico e, se mantida, permitiria ao modelo aprender ruído associado à ordem de coleta.

A variável alvo `diagnosis` é categórica (`M`/`B`) e foi codificada como **M = 1** e **B = 0**.
Maligno é a classe positiva por ser o evento que se deseja detectar e cujo erro tem maior custo
clínico.

### 2.2 Distribuição das classes

![Distribuição dos diagnósticos](../results/figures/01_balanceamento_classes.png)

| Classe | Amostras | Proporção |
|---|---|---|
| Benigno | 357 | 62,7% |
| Maligno | 212 | 37,3% |

O desbalanceamento é leve e não exige reamostragem, mas tem duas consequências diretas:

1. **A acurácia isolada engana.** Um classificador que respondesse "benigno" para toda paciente já
   atingiria 62,7% de acurácia. Esse é o piso contra o qual qualquer resultado precisa ser
   comparado.
2. **A divisão treino/teste precisa ser estratificada**, preservando a proporção das classes nos
   dois conjuntos para que a avaliação continue representativa.

### 2.3 Escalas e assimetria

Duas características das features condicionam todo o pré-processamento:

- **Escalas radicalmente diferentes.** `area_worst` varia por milhares de unidades, enquanto
  `fractal_dimension_se` varia por centésimos — cerca de cinco ordens de grandeza de diferença.
  Algoritmos baseados em distância (KNN) ou em otimização com regularização (regressão logística)
  seriam dominados pelas variáveis de maior magnitude.
- **Assimetria à direita.** Várias features, principalmente as do grupo `_se`, apresentam cauda
  longa à direita: maioria das amostras em valores baixos e poucos casos com valores muito altos.
  Isso é esperado em medidas biológicas de tamanho e irregularidade.

### 2.4 Distribuições por diagnóstico

![Distribuição das medidas médias](../results/figures/02_distribuicoes_mean.png)

As medidas de tamanho e de irregularidade — `radius_mean`, `perimeter_mean`, `area_mean`,
`concavity_mean`, `concave points_mean` — mostram tumores malignos deslocados para valores mais
altos, com sobreposição apenas parcial entre as classes.

Já `smoothness_mean`, `symmetry_mean` e `fractal_dimension_mean` têm distribuições quase
sobrepostas: isoladamente, quase não distinguem os grupos.

![Dispersão das medidas worst](../results/figures/03_boxplots_worst.png)

O grupo `_worst` separa as classes com mais nitidez que o grupo `_mean`. O achado tem sentido
clínico: o que caracteriza malignidade não é o comportamento médio do tecido, mas a existência de
células com morfologia mais agressiva — a "pior" célula da amostra é mais informativa que a célula
típica.

Os pontos fora dos bigodes são numerosos, mas **não são erros de medição**: são casos extremos
reais, concentrados na classe maligna. Removê-los descartaria justamente os casos mais graves,
exatamente os que o modelo precisa acertar. **Nenhum outlier foi removido neste projeto.**

### 2.5 Padrões relacionados ao diagnóstico

![Top 15 features por poder de separação](../results/figures/04_separacao_features.png)

O *d* de Cohen mede a distância entre as médias das duas classes em unidades de desvio padrão. Por
ser adimensional, permite comparar features de escalas muito diferentes. Valores acima de 0,8 já
indicam efeito grande.

| Feature | Média benigno | Média maligno | *d* de Cohen |
|---|---|---|---|
| concave points_worst | 0,0744 | 0,1822 | 2,69 |
| perimeter_worst | 87,01 | 141,37 | 2,60 |
| concave points_mean | 0,0257 | 0,0880 | 2,54 |
| radius_worst | 13,38 | 21,13 | 2,54 |
| perimeter_mean | 78,08 | 115,37 | 2,29 |

Padrões identificados:

- **Concavidade e pontos côncavos lideram o ranking.** Medem o quanto o contorno do núcleo é
  irregular e reentrante — assinatura morfológica clássica de células malignas.
- **Medidas de tamanho vêm em seguida:** núcleos malignos são maiores.
- **Todos os *d* de Cohen do topo são positivos**, ou seja, os valores são sistematicamente maiores
  nos tumores malignos, sem inversão de sinal entre as principais features.
- **Textura, simetria e dimensão fractal** ficam no fim do ranking, com poder de separação fraco
  isoladamente — o que não impede contribuição em combinação com as demais.

Raio, perímetro e área descrevem essencialmente a mesma geometria, o que antecipa forte
multicolinearidade — quantificada na seção 3.

### 2.6 Síntese

| Achado | Consequência |
|---|---|
| Nenhum valor ausente real; só a coluna fantasma do CSV | Limpeza estrutural, sem imputação |
| `id` é identificador sem valor preditivo | Removida |
| Alvo categórico `M`/`B` | Codificado como M = 1 (positivo) e B = 0 |
| 62,7% benignos / 37,3% malignos | Acurácia isolada insuficiente; split estratificado |
| Escalas com até cinco ordens de grandeza de diferença | `StandardScaler` dentro do pipeline |
| Assimetria à direita, muitos outliers legítimos | Outliers preservados |
| Grupo `_worst` separa melhor que `_mean` | Confirmado depois pela importância dos modelos |
| Concavidade e tamanho lideram a separação | Coerente com a morfologia de células malignas |
| Raio, perímetro e área medem a mesma geometria | Multicolinearidade analisada na seção 3 |

---

## 3. Estratégias de pré-processamento

Código correspondente: [`src/preprocessing.py`](../src/preprocessing.py)

### 3.1 Limpeza aplicada

| Problema | Tratamento | Justificativa |
|---|---|---|
| Coluna `Unnamed: 32`, totalmente nula | Removida | Artefato da vírgula final no cabeçalho do CSV, não é dado faltante |
| Coluna `id` | Removida | Identificador do paciente, sem relação causal com o diagnóstico |
| Alvo `diagnosis` categórico (`M`/`B`) | Codificado como M = 1, B = 0 | Maligno é a classe positiva: é o evento a detectar |
| Outliers nas medidas morfológicas | **Preservados** | São casos malignos extremos reais, não erros de medição |
| Valores ausentes | Nenhum na base | Imputador de mediana mantido no pipeline por robustez |

A decisão de preservar outliers merece destaque: em uma base clínica, os valores extremos são
frequentemente os casos mais graves. Removê-los melhoraria as métricas de forma enganosa,
eliminando justamente as pacientes que o sistema mais precisa identificar.

### 3.2 Conversão de variáveis

As 30 features preditoras já são numéricas contínuas e não exigem codificação — não há variáveis
categóricas entre elas. A única variável categórica da base é o alvo, convertido para binário na
etapa de limpeza (`src/data.py`).

O que essas variáveis exigem não é codificação, e sim **padronização**, pela diferença de escala
descrita na seção 2.3.

### 3.3 Pipeline de pré-processamento

```
Pipeline
  ├── preprocessor
  │     ├── SimpleImputer(strategy="median")   # defensivo
  │     └── StandardScaler()                   # média 0, desvio padrão 1
  └── estimator                                # modelo da vez
```

**Por que o scaler fica dentro do pipeline.** Se fosse ajustado sobre a base completa antes da
separação treino/teste, a média e o desvio usados na transformação carregariam informação do
conjunto de teste — vazamento de dados (*data leakage*). O efeito é uma métrica otimista que não se
sustenta em produção. Dentro do `Pipeline`, o scaler é reajustado a cada fold da validação cruzada,
usando apenas os dados de treino daquele fold.

Efeito prático da padronização, no conjunto de treino:

| Feature | Média antes | Desvio antes | Média depois | Desvio depois |
|---|---|---|---|---|
| `area_worst` | 890,57 | 582,35 | 0 | 1 |
| `fractal_dimension_se` | 0,00377 | 0,00263 | 0 | 1 |

O imputador de mediana é preventivo: a base atual não tem valores ausentes, mas o pipeline
permanece válido caso novos dados cheguem incompletos, e a mediana é robusta à assimetria
observada na seção 2.3.

### 3.4 Análise de correlação

![Correlação entre as features](../results/figures/05_correlacao.png)

O mapa é dominado por correlações positivas: praticamente todas as features medem aspectos do
tamanho e da irregularidade do mesmo núcleo celular. As poucas correlações negativas relevantes
envolvem `fractal_dimension_mean`, que se comporta de forma inversa às medidas de tamanho.

**Multicolinearidade.** Existem **21 pares de features com correlação acima de 0,9**, e os extremos
são quase perfeitos:

| Par | Correlação |
|---|---|
| `radius_mean` × `perimeter_mean` | 0,998 |
| `radius_worst` × `perimeter_worst` | 0,994 |
| `radius_mean` × `area_mean` | 0,987 |
| `perimeter_mean` × `area_mean` | 0,987 |
| `radius_worst` × `area_worst` | 0,984 |

Isso não é acaso estatístico, é **geometria**: em uma forma aproximadamente circular, perímetro e
área são funções diretas do raio. As três variáveis medem a mesma grandeza em unidades diferentes.

Consequências por família de modelo:

- **Regressão logística** — a capacidade preditiva não é prejudicada, mas os coeficientes
  individuais ficam instáveis: o peso se distribui de forma arbitrária entre variáveis redundantes.
  A regularização L2 (padrão do scikit-learn) atenua o problema ao distribuir o peso de maneira
  estável, mas a *interpretação* de cada coeficiente isolado exige cautela.
- **KNN** — a mesma informação é contada várias vezes no cálculo da distância, dando peso excessivo
  à dimensão "tamanho do núcleo".
- **Random Forest** — robusta na predição, mas a importância nativa fica diluída: features
  redundantes dividem o crédito entre si e nenhuma parece tão relevante quanto de fato é. Esse é um
  dos motivos para complementar a análise com SHAP na seção 5.

**Decisão: nenhuma feature foi removida.** Com 569 amostras e 30 features, a base não sofre de alta
dimensionalidade, e a regularização já lida com a redundância. Seleção manual introduziria uma
escolha arbitrária sem ganho esperado de desempenho. A multicolinearidade fica registrada para
orientar a leitura dos coeficientes.

**Correlação com o alvo:**

![Correlação com o diagnóstico](../results/figures/06_correlacao_alvo.png)

Como o alvo é binário, a correlação de Pearson equivale ao coeficiente ponto-bisserial. O ranking
reproduz quase exatamente o do *d* de Cohen — `concave points_worst` (0,79), `perimeter_worst`
(0,78) e `concave points_mean` (0,78) no topo. Duas métricas independentes chegando ao mesmo
resultado reforçam o achado.

Nenhuma feature isolada chega perto de correlação 1 com o diagnóstico: **não existe atalho de
variável única**, a classificação depende da combinação de várias medidas.

### 3.5 Separação treino/teste

Divisão de 80/20 com `stratify=y` e `random_state=42`:

| Conjunto | Amostras | Benigno | Maligno | Proporção maligno |
|---|---|---|---|---|
| Treino | 455 | 285 | 170 | 37,4% |
| Teste | 114 | 72 | 42 | 36,8% |
| *Base completa* | *569* | *357* | *212* | *37,3%* |

A estratificação mantém a proporção original nos dois conjuntos. Sem ela, a variação em uma base
deste tamanho poderia tornar o conjunto de teste não representativo.

O conjunto de teste permanece intocado até a avaliação final: toda a comparação entre modelos é
feita por validação cruzada estratificada de 5 folds **dentro do conjunto de treino**.

---

## 4. Modelos utilizados e justificativa

Código correspondente: [`src/models.py`](../src/models.py) ·
Notebook: [`notebooks/02_modeling_evaluation.ipynb`](../notebooks/02_modeling_evaluation.ipynb)

### 4.1 Modelos escolhidos

Foram avaliadas três técnicas com fundamentos distintos, para que a comparação seja informativa e
não apenas uma variação do mesmo viés:

| Modelo | Como decide | Por que foi incluído |
|---|---|---|
| **Regressão Logística** | combinação linear das features passada por uma sigmoide | baseline interpretável — cada coeficiente indica quanto uma medida empurra a previsão, o que é essencial em contexto clínico |
| **KNN** | classe majoritária entre os *k* casos mais parecidos | não paramétrico, não assume forma da fronteira de decisão; depende fortemente de escala, o que evidencia o efeito da padronização |
| **Random Forest** | votação de muitas árvores treinadas em amostras diferentes | captura relações não lineares e interações entre medidas; fornece importância de variáveis nativa |

Todos são `Pipeline` completos (pré-processamento + estimador), o que garante que a padronização
seja reajustada dentro de cada fold da validação cruzada.

### 4.2 Escolha da métrica

O problema é clínico e assimétrico: **os dois tipos de erro não custam a mesma coisa.**

| Erro | Significado clínico | Consequência |
|---|---|---|
| **Falso negativo** | tumor maligno classificado como benigno | paciente vai para casa sem tratamento; o diagnóstico atrasa |
| **Falso positivo** | tumor benigno classificado como maligno | exames adicionais, ansiedade, custo |

Um falso negativo é incomparavelmente mais grave. Por isso a **métrica prioritária é o recall da
classe maligna** — a fração de tumores malignos que o modelo consegue capturar.

Duas ressalvas definem o protocolo adotado:

- **A acurácia não decide.** Com 62,7% de casos benignos, um classificador que respondesse "benigno"
  sempre atingiria 62,7% de acurácia com recall zero. A acurácia é reportada, nunca isolada.
- **Recall puro também não decide a escolha do modelo.** Otimizar apenas recall premiaria um
  classificador que chama quase tudo de maligno — recall 100%, precisão baixa, inútil na triagem.
  Por isso a *seleção* do modelo usa **F1** (média harmônica entre precisão e recall), e o recall é
  otimizado depois, pelo ajuste do limiar de decisão (seção 5.4).

### 4.3 Protocolo de validação

- Divisão estratificada 80/20, com o conjunto de teste **intocado** até a avaliação final
- Comparação entre modelos por **validação cruzada estratificada de 5 folds dentro do treino**
- Ajuste de hiperparâmetros por busca em grade pequena, otimizando F1 na mesma validação cruzada
- Semente fixa (`random_state=42`) em todas as etapas, para reprodutibilidade

Resultados da validação cruzada (conjunto de treino, 455 amostras):

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Regressão Logística** | 0,974 | 0,977 | **0,953** | **0,964** | 0,996 |
| KNN | 0,965 | 0,988 | 0,918 | 0,951 | 0,987 |
| Random Forest | 0,960 | 0,958 | 0,935 | 0,946 | 0,988 |

A regressão logística lidera em F1 e recall, com desvio padrão baixo entre os folds — resultado
estável, não fruto de uma partição favorável. O KNN tem a maior precisão (0,988) e o menor recall
(0,918): é conservador, erra pouco quando afirma "maligno", mas deixa passar mais casos malignos.
Para este problema, é a troca errada.

**Um modelo linear vencendo dois não lineares é informativo:** indica que, após a padronização, as
classes são quase linearmente separáveis no espaço das 30 features. Isso é coerente com a análise
exploratória, que mostrou várias medidas fortemente deslocadas entre os grupos e nenhuma interação
complexa aparente.

Ajuste de hiperparâmetros:

| Modelo | Melhores parâmetros | F1 na validação |
|---|---|---|
| Regressão Logística | `C = 1,0` | 0,964 |
| KNN | `n_neighbors = 3`, `weights = uniform` | 0,963 |
| Random Forest | `max_depth = 6`, `min_samples_leaf = 1`, `n_estimators = 200` | 0,952 |

O ajuste altera pouco: a regressão logística já estava no `C` padrão; o KNN melhora ao reduzir *k*
de 5 para 3; a Random Forest melhora ao limitar a profundidade, sinal de leve sobreajuste. Nenhum
ganho muda a ordem entre os modelos, o que reforça que a diferença vem da natureza de cada
algoritmo, e não de configuração.

**Modelo escolhido: Regressão Logística.**

---

## 5. Resultados e interpretação

### 5.1 Desempenho no conjunto de teste

Primeiro e único uso dos 114 casos separados no início.

![Comparação dos modelos](../results/figures/09_comparacao_modelos.png)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| **Regressão Logística** | 0,965 | 0,975 | **0,929** | **0,951** | **0,996** | **3** |
| Random Forest | 0,965 | 1,000 | 0,905 | 0,950 | 0,995 | 4 |
| KNN | 0,939 | 0,973 | 0,857 | 0,911 | 0,983 | 6 |

O desempenho no teste confirma o da validação cruzada, sem queda relevante — não há sinal de
sobreajuste.

A Random Forest alcança precisão perfeita (nenhum falso positivo), mas com 4 falsos negativos.
**Trocar um falso positivo por um falso negativo é um mau negócio neste contexto:** o primeiro custa
um exame adicional, o segundo custa um diagnóstico perdido.

### 5.2 Matriz de confusão

![Matriz de confusão](../results/figures/07_matriz_confusao.png)

O modelo escolhido acerta 110 dos 114 casos:

| Quadrante | Casos | Leitura clínica |
|---|---|---|
| Verdadeiros negativos | 71 | tumores benignos corretamente identificados |
| Verdadeiros positivos | 39 | tumores malignos corretamente identificados |
| Falsos positivos | 1 | uma paciente encaminhada a exames adicionais desnecessários |
| **Falsos negativos** | **3** | **três tumores malignos classificados como benignos** |

Os três falsos negativos são o número que importa. Com recall de 0,929, o modelo deixa passar cerca
de 1 em cada 14 tumores malignos — bom para um sistema de apoio, **inaceitável para decisão
autônoma**.

### 5.3 Curvas ROC

![Curvas ROC](../results/figures/08_curvas_roc.png)

As três curvas ficam próximas do canto superior esquerdo, com AUC entre 0,983 e 0,996. A AUC mede a
capacidade de ordenar corretamente os casos por risco, independentemente do limiar.

AUC de 0,996 combinada a recall de 0,929 no limiar padrão revela algo importante: **o modelo separa
bem as classes; o que está mal calibrado é o corte.**

### 5.4 Ajuste do limiar de decisão

![Efeito do limiar](../results/figures/10_limiar_decisao.png)

O limiar de 0,5 é apenas a convenção do `predict()`, não uma escolha clínica. Baixá-lo para **0,30**
melhora todas as métricas simultaneamente:

| Limiar | Recall | Precisão | F1 | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|
| 0,50 (padrão) | 0,929 | 0,975 | 0,951 | 3 | 1 |
| **0,30** | **0,976** | **0,976** | **0,976** | **1** | 1 |

Em termos clínicos: duas pacientes a mais com tumor maligno seriam corretamente sinalizadas, sem
aumento de exames desnecessários.

Duas ressalvas honestas sobre esse resultado:

1. **O conjunto de teste tem 114 casos.** Uma diferença de dois falsos negativos é estatisticamente
   frágil e pode não se repetir em outra amostra. O limiar ideal deveria ser calibrado por validação
   cruzada no treino, e não escolhido observando o teste — caso contrário, ajusta-se ao próprio
   conjunto de avaliação.
2. **A direção do ajuste é defensável independentemente do valor exato.** Como o custo de um falso
   negativo é muito maior que o de um falso positivo, um limiar abaixo de 0,5 é a escolha racional
   para triagem, ainda que o ótimo preciso varie.

### 5.5 Explicabilidade

*A ser preenchida na Fase 5: coeficientes da regressão logística, importância por permutação, SHAP
e interpretação clínica das previsões individuais.*

---

## 6. Discussão crítica e uso na prática

*A ser preenchida na Fase 7: limitações da base e do modelo, cenários de uso real, riscos e o papel
final do médico na decisão.*

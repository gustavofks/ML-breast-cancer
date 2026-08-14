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

*A ser preenchida na Fase 3: limpeza aplicada, conversão de variáveis, pipeline de
pré-processamento, análise de correlação e multicolinearidade, separação treino/teste.*

---

## 4. Modelos utilizados e justificativa

*A ser preenchida na Fase 4: modelos escolhidos, motivo de cada escolha, protocolo de validação
cruzada, escolha e justificativa da métrica prioritária.*

---

## 5. Resultados e interpretação

*A ser preenchida nas Fases 4 e 5: métricas no conjunto de teste, matriz de confusão, curva ROC,
comparação entre modelos, feature importance, SHAP e interpretação clínica dos resultados.*

---

## 6. Discussão crítica e uso na prática

*A ser preenchida na Fase 7: limitações da base e do modelo, cenários de uso real, riscos e o papel
final do médico na decisão.*

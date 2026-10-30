"""Custo clinico do individuo, medido por validacao cruzada dentro do treino.

Duas regras metodologicas estao codificadas aqui, e nenhuma delas e negociavel.

**O fitness so ve treino.** A funcao recebe `(X, y)` por parametro, e o pacote
inteiro nao importa nada que saiba abrir a base ou refazer a separacao. A Fase 1
escolheu o limiar 0,30 olhando o conjunto de teste e classificou isso como
fragilidade metodologica; e essa fragilidade que este modulo corrige.

**O limiar e do individuo.** Cada candidato aplica o proprio corte sobre a
probabilidade prevista fora de fold, entao o limiar e otimizado junto com o
resto, e nao escolhido depois olhando o resultado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.genetic.chromosome import Individuo, decodificar


@dataclass(frozen=True)
class CustoClinico:
    """Politica de custo. Os defaults vem de `config`, nunca do corpo do codigo.

    A varredura de sensibilidade da secao 4.5.1 do design existe justamente para
    variar estes tres numeros e mostrar como a solucao se desloca — o que
    transforma a razao de custo numa decisao informada da instituicao de saude,
    em vez de uma constante arbitraria do cientista de dados.
    """

    falso_negativo: float = config.CUSTO_FALSO_NEGATIVO
    falso_positivo: float = config.CUSTO_FALSO_POSITIVO
    parcimonia: float = config.PENALIDADE_PARCIMONIA


@dataclass(frozen=True)
class Avaliacao:
    """Resultado de uma avaliacao completa, guardado no historico da execucao."""

    fitness: float
    custo: float
    recall: float
    falsos_negativos: int
    falsos_positivos: int
    n_amostras: int
    n_features: int


def custo_medio(
    *,
    falsos_negativos: int,
    falsos_positivos: int,
    n: int,
    custo: CustoClinico,
) -> float:
    """Custo clinico por amostra."""
    total = custo.falso_negativo * falsos_negativos + custo.falso_positivo * falsos_positivos
    return float(total / n)


def combinar(
    *,
    custo_medio: float,
    n_features: int,
    total_features: int,
    custo: CustoClinico,
) -> float:
    """Junta custo e parcimonia num unico numero a maximizar.

    A normalizacao divide pelo custo do falso negativo porque esse e o pior custo
    possivel de uma amostra: o termo fica em [0, 1] seja qual for a razao
    escolhida, e fitness de razoes diferentes continua comparavel em escala.
    """
    normalizado = custo_medio / custo.falso_negativo
    return float(1.0 - normalizado - custo.parcimonia * (n_features / total_features))


def avaliar(
    individuo: Individuo,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    custo: CustoClinico | None = None,
    cv_seed: int = config.SEED,
    folds: int = config.CV_FOLDS,
) -> Avaliacao:
    """Fitness do individuo por validacao cruzada estratificada em `(X, y)`.

    `X` e `y` sao sempre o conjunto de treino. As probabilidades sao acumuladas
    fora de fold, entao cada linha e prevista exatamente uma vez por um modelo
    que nao a viu, e a matriz de confusao final usa todas as linhas recebidas.
    """
    custo = custo or CustoClinico()
    pipeline, selecionadas = decodificar(individuo, list(X.columns))
    X_ativas = X.loc[:, selecionadas]

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    probabilidades = np.zeros(len(y), dtype=float)

    for indices_ajuste, indices_previsao in cv.split(X_ativas, y):
        modelo = clone(pipeline)
        modelo.fit(X_ativas.iloc[indices_ajuste], y.iloc[indices_ajuste])
        probabilidades[indices_previsao] = modelo.predict_proba(
            X_ativas.iloc[indices_previsao]
        )[:, 1]

    previsoes = (probabilidades >= individuo.limiar).astype(int)
    _, fp, fn, _ = confusion_matrix(y, previsoes, labels=[0, 1]).ravel()

    medio = custo_medio(
        falsos_negativos=int(fn), falsos_positivos=int(fp), n=len(y), custo=custo
    )
    valor = combinar(
        custo_medio=medio,
        n_features=len(selecionadas),
        total_features=X.shape[1],
        custo=custo,
    )

    return Avaliacao(
        fitness=valor,
        custo=medio,
        recall=float(recall_score(y, previsoes, zero_division=0)),
        falsos_negativos=int(fn),
        falsos_positivos=int(fp),
        n_amostras=int(len(y)),
        n_features=len(selecionadas),
    )


@dataclass(frozen=True)
class Estabilidade:
    """Comparacao entre o fitness da busca e o de outra particao.

    `queda` positiva significa piora, e e o sinal de sobreajuste as particoes
    especificas que a busca otimizou. Centenas de avaliacoes sobre as mesmas
    cinco dobras bastam para que o algoritmo aprenda a particao, nao o problema,
    e a secao 4.7 do design exige medir isso em vez de supor que nao aconteceu.
    """

    fitness_original: float
    fitness_revalidado: float
    queda: float


def reavaliar_estabilidade(
    individuo: Individuo,
    avaliacao: Avaliacao,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    custo: CustoClinico | None = None,
    cv_seed: int,
) -> Estabilidade:
    """Reavalia o individuo com outra semente de validacao cruzada.

    `cv_seed` e obrigatorio e nao tem default de proposito: reavaliar com a
    mesma semente da busca nao mede nada, e um default silencioso convidaria
    exatamente esse erro.
    """
    revalidada = avaliar(individuo, X, y, custo=custo, cv_seed=cv_seed)
    return Estabilidade(
        fitness_original=avaliacao.fitness,
        fitness_revalidado=revalidada.fitness,
        queda=avaliacao.fitness - revalidada.fitness,
    )


def reavaliar_lote(
    melhores: list[tuple[Individuo, Avaliacao]],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    custo: CustoClinico | None = None,
    cv_seed: int,
) -> list[Estabilidade]:
    """Reavalia os cinco melhores de uma execucao, preservando a ordem."""
    return [
        reavaliar_estabilidade(individuo, avaliacao, X, y, custo=custo, cv_seed=cv_seed)
        for individuo, avaliacao in melhores
    ]

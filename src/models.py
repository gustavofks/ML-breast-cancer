"""Definicao, ajuste e comparacao dos modelos de classificacao.

Todos os modelos sao `Pipeline` completos (pre-processamento + estimador),
criados por `src.preprocessing.build_pipeline`. Isso garante que a padronizacao
seja reajustada dentro de cada fold da validacao cruzada, sem vazamento.

A comparacao entre modelos e feita **apenas com o conjunto de treino**, por
validacao cruzada estratificada de 5 folds. O conjunto de teste so e usado na
avaliacao final (`src/evaluation.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_pipeline

# Metricas acompanhadas em toda comparacao. `recall` e a prioritaria do problema
# (falso negativo = tumor maligno nao detectado); `f1` e usada para *escolher* o
# modelo, por equilibrar recall e precisao — otimizar recall puro levaria a um
# classificador que chama quase tudo de maligno.
SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")
SELECTION_METRIC = "f1"

# Grades pequenas de hiperparametros: a base tem 569 amostras, entao a busca e
# barata, mas grades grandes so aumentariam o risco de sobreajuste a validacao.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "Regressão Logística": {"estimator__C": [0.01, 0.1, 1.0, 10.0]},
    "KNN": {
        "estimator__n_neighbors": [3, 5, 7, 9, 11],
        "estimator__weights": ["uniform", "distance"],
    },
    "Random Forest": {
        "estimator__n_estimators": [200, 400],
        "estimator__max_depth": [None, 6],
        "estimator__min_samples_leaf": [1, 2],
    },
}


def get_models() -> dict[str, Pipeline]:
    """Catalogo de modelos avaliados no projeto.

    Tres tecnicas com fundamentos diferentes, para que a comparacao seja
    informativa e nao apenas uma variacao do mesmo vies:

    - **Regressão Logística**: baseline linear e interpretavel; os coeficientes
      dizem diretamente como cada medida empurra a previsao.
    - **KNN**: nao parametrico, classifica por semelhanca com casos conhecidos.
      Depende fortemente de escala, o que evidencia o efeito da padronizacao.
    - **Random Forest**: conjunto de arvores, captura relacoes nao lineares e
      interacoes entre medidas, e fornece importancia de variaveis nativa.
    """
    return {
        "Regressão Logística": build_pipeline(
            LogisticRegression(max_iter=5000, random_state=config.SEED)
        ),
        "KNN": build_pipeline(KNeighborsClassifier()),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1)
        ),
    }


def _cv() -> StratifiedKFold:
    """Validacao cruzada estratificada, embaralhada e reprodutivel."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def cross_validate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Compara os modelos por validacao cruzada no conjunto de treino.

    Returns:
        Uma linha por modelo, com media e desvio padrao de cada metrica entre os
        folds, ordenada pela metrica de selecao.
    """
    linhas = []
    for nome, modelo in models.items():
        resultado = cross_validate(
            modelo,
            X_train,
            y_train,
            cv=_cv(),
            scoring=list(SCORING),
            n_jobs=-1,
        )
        linha = {"modelo": nome}
        for metrica in SCORING:
            valores = resultado[f"test_{metrica}"]
            linha[metrica] = round(float(np.mean(valores)), 4)
            linha[f"{metrica}_std"] = round(float(np.std(valores)), 4)
        linhas.append(linha)

    return (
        pd.DataFrame(linhas)
        .sort_values(SELECTION_METRIC, ascending=False)
        .reset_index(drop=True)
    )


def tune_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Ajusta hiperparametros de cada modelo por busca em grade.

    A busca usa a mesma validacao cruzada estratificada e otimiza a metrica de
    selecao (F1). Modelos sem grade definida sao devolvidos sem alteracao.

    Returns:
        `(modelos_ajustados, tabela_de_resultados)`. Os modelos devolvidos ja
        estao treinados no conjunto de treino completo com os melhores
        parametros (`refit` do `GridSearchCV`).
    """
    ajustados: dict[str, Pipeline] = {}
    linhas = []

    for nome, modelo in models.items():
        grade = PARAM_GRIDS.get(nome)
        if not grade:
            ajustados[nome] = modelo.fit(X_train, y_train)
            continue

        busca = GridSearchCV(
            modelo,
            param_grid=grade,
            scoring=SELECTION_METRIC,
            cv=_cv(),
            n_jobs=-1,
        )
        busca.fit(X_train, y_train)

        ajustados[nome] = busca.best_estimator_
        linhas.append(
            {
                "modelo": nome,
                "melhores_parametros": {
                    chave.replace("estimator__", ""): valor
                    for chave, valor in busca.best_params_.items()
                },
                f"{SELECTION_METRIC}_validacao": round(float(busca.best_score_), 4),
            }
        )

    tabela = pd.DataFrame(linhas).sort_values(
        f"{SELECTION_METRIC}_validacao", ascending=False
    ).reset_index(drop=True)
    return ajustados, tabela


def select_best(cv_results: pd.DataFrame, metric: str = SELECTION_METRIC) -> str:
    """Nome do modelo com melhor desempenho na metrica indicada."""
    if metric not in cv_results.columns:
        raise ValueError(f"Metrica '{metric}' ausente na tabela de resultados.")
    return str(cv_results.loc[cv_results[metric].idxmax(), "modelo"])


def fit_final(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Treina o modelo escolhido com todo o conjunto de treino."""
    return model.fit(X_train, y_train)

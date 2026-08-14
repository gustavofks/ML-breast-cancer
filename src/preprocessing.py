"""Pipeline de pre-processamento e separacao treino/teste.

Duas decisoes centrais estao codificadas aqui:

1. **A padronizacao vive dentro do `Pipeline`.** Se o `StandardScaler` fosse
   ajustado sobre a base inteira antes do split, a media e o desvio usados na
   transformacao carregariam informacao do conjunto de teste — vazamento de dados
   (*data leakage*), que infla artificialmente as metricas. Dentro do pipeline, o
   scaler e reajustado a cada fold da validacao cruzada, usando apenas dados de
   treino daquele fold.

2. **O split e estratificado.** A base tem 62,7% de casos benignos; sem
   estratificacao, treino e teste poderiam receber proporcoes diferentes e a
   avaliacao deixaria de ser representativa.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def build_preprocessor() -> Pipeline:
    """Monta o pre-processador aplicado a todas as features numericas.

    Etapas:
        - `SimpleImputer(strategy="median")`: defensivo. A base atual nao tem
          valores ausentes, mas a mediana e robusta a assimetria e mantem o
          pipeline valido caso novos dados cheguem incompletos.
        - `StandardScaler`: centra em zero e escala para desvio padrao 1. As
          features variam por ate cinco ordens de grandeza entre si; sem isso,
          KNN e regressao logistica regularizada seriam dominados pelas
          variaveis de maior magnitude.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Encadeia o pre-processamento a um estimador.

    Todo modelo do projeto e criado por esta funcao, entao nenhum codigo externo
    precisa lembrar de padronizar os dados antes de treinar ou prever.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("estimator", estimator),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste preservando a proporcao das classes.

    Returns:
        `(X_train, X_test, y_train, y_test)`.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def split_summary(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Tabela comparando o tamanho e a composicao dos dois conjuntos."""
    rows = []
    for nome, alvo in (("Treino", y_train), ("Teste", y_test)):
        rows.append(
            {
                "conjunto": nome,
                "amostras": len(alvo),
                "benigno": int((alvo == 0).sum()),
                "maligno": int((alvo == 1).sum()),
                "proporcao_maligno": round(float((alvo == 1).mean()), 4),
            }
        )
    return pd.DataFrame(rows)

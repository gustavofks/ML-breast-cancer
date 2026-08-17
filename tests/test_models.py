"""Testes do catalogo de modelos e da comparacao por validacao cruzada."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import models
from src.data import load_dataset
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def treino():
    X, y = load_dataset()
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train


def test_catalogo_traz_tres_pipelines_completos():
    catalogo = models.get_models()

    assert set(catalogo) == {"Regressão Logística", "KNN", "Random Forest"}
    for pipeline in catalogo.values():
        assert isinstance(pipeline, Pipeline)
        # Todo modelo carrega o proprio pre-processamento: nenhum codigo externo
        # precisa lembrar de padronizar antes de treinar.
        assert list(pipeline.named_steps) == ["preprocessor", "estimator"]


def test_validacao_cruzada_traz_todas_as_metricas_por_modelo(treino):
    X_train, y_train = treino
    catalogo = {"Regressão Logística": models.get_models()["Regressão Logística"]}

    resultados = models.cross_validate_models(catalogo, X_train, y_train)

    assert len(resultados) == 1
    for metrica in models.SCORING:
        assert metrica in resultados.columns
        assert f"{metrica}_std" in resultados.columns
        assert 0.0 <= resultados.iloc[0][metrica] <= 1.0


def test_select_best_escolhe_o_maior_valor_da_metrica():
    tabela = pd.DataFrame(
        {"modelo": ["A", "B", "C"], "f1": [0.80, 0.95, 0.91], "recall": [0.99, 0.90, 0.92]}
    )

    assert models.select_best(tabela) == "B"
    assert models.select_best(tabela, metric="recall") == "A"


def test_select_best_rejeita_metrica_inexistente():
    tabela = pd.DataFrame({"modelo": ["A"], "f1": [0.9]})

    with pytest.raises(ValueError, match="ausente"):
        models.select_best(tabela, metric="auc_inexistente")


def test_metrica_de_selecao_e_f1_e_nao_recall_puro():
    """Otimizar recall isolado premiaria um modelo que chama tudo de maligno."""
    assert models.SELECTION_METRIC == "f1"
    assert "recall" in models.SCORING

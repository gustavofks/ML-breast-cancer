"""Testes das tecnicas de explicabilidade."""

import numpy as np
import pytest

from src import explain, models
from src.data import load_dataset
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def contexto():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    catalogo = models.get_models()
    linear = catalogo["Regressão Logística"].fit(X_train, y_train)
    floresta = catalogo["Random Forest"].fit(X_train, y_train)
    return linear, floresta, X_train, X_test, y_test


def test_coeficientes_cobrem_todas_as_features_e_vem_ordenados(contexto):
    linear, _, X_train, _, _ = contexto
    coeficientes = explain.linear_coefficients(linear, list(X_train.columns))

    assert len(coeficientes) == len(X_train.columns)
    assert coeficientes["coeficiente"].abs().is_monotonic_decreasing
    # A razao de chances e a exponencial do coeficiente.
    assert np.allclose(
        coeficientes["razao_de_chances"],
        np.exp(coeficientes["coeficiente"]).round(3),
        atol=1e-3,
    )


def test_coeficientes_rejeitam_modelo_sem_coef(contexto):
    _, floresta, X_train, _, _ = contexto

    with pytest.raises(TypeError, match="não é um modelo linear"):
        explain.linear_coefficients(floresta, list(X_train.columns))


def test_importancia_de_arvore_soma_um(contexto):
    _, floresta, X_train, _, _ = contexto
    importancias = explain.tree_importances(floresta, list(X_train.columns))

    assert len(importancias) == len(X_train.columns)
    assert abs(importancias["importancia"].sum() - 1.0) < 0.01
    assert importancias["importancia"].is_monotonic_decreasing


def test_permutacao_avalia_todas_as_features(contexto):
    linear, _, _, X_test, y_test = contexto
    # n_repeats baixo mantem o teste rapido; a ordenacao ja e verificavel.
    permutacao = explain.permutation_scores(linear, X_test, y_test, n_repeats=5)

    assert len(permutacao) == len(X_test.columns)
    assert permutacao["queda_media"].is_monotonic_decreasing
    assert (permutacao["desvio"] >= 0).all()


def test_shap_produz_um_valor_por_feature_e_por_amostra(contexto):
    if not explain.shap_available():
        pytest.skip("SHAP indisponível neste ambiente (numba bloqueado pelo sistema)")

    linear, _, X_train, X_test, _ = contexto
    explicacao = explain.shap_explanation(linear, X_train, X_test)

    assert explicacao.values.shape == (len(X_test), len(X_test.columns))

    importancia = explain.shap_importance(explicacao)
    assert len(importancia) == len(X_test.columns)
    assert importancia["shap_medio"].is_monotonic_decreasing
    assert (importancia["shap_medio"] >= 0).all()

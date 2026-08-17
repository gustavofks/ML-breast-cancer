"""Testes das metricas de avaliacao e da analise de limiar."""

import json

import pytest

from src import evaluation, models
from src.data import load_dataset
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def modelo_e_teste():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    modelo = models.get_models()["Regressão Logística"].fit(X_train, y_train)
    return modelo, X_test, y_test


def test_evaluate_traz_metricas_e_quadrantes_consistentes(modelo_e_teste):
    modelo, X_test, y_test = modelo_e_teste
    metricas = evaluation.evaluate(modelo, X_test, y_test)

    quadrantes = [
        "verdadeiros_negativos",
        "falsos_positivos",
        "falsos_negativos",
        "verdadeiros_positivos",
    ]
    assert sum(metricas[chave] for chave in quadrantes) == len(y_test)
    for metrica in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metricas[metrica] <= 1.0


def test_recall_bate_com_a_contagem_de_falsos_negativos(modelo_e_teste):
    modelo, X_test, y_test = modelo_e_teste
    m = evaluation.evaluate(modelo, X_test, y_test)

    malignos = m["verdadeiros_positivos"] + m["falsos_negativos"]
    assert round(m["verdadeiros_positivos"] / malignos, 4) == m["recall"]


def test_limiar_menor_nunca_perde_recall(modelo_e_teste):
    """Abaixar o limiar so pode manter ou aumentar o recall."""
    modelo, X_test, y_test = modelo_e_teste
    tabela = evaluation.threshold_analysis(modelo, X_test, y_test)

    assert tabela["recall"].is_monotonic_decreasing
    assert tabela["falsos_negativos"].is_monotonic_increasing


def test_evaluate_all_ordena_por_recall(modelo_e_teste):
    modelo, X_test, y_test = modelo_e_teste
    tabela = evaluation.evaluate_all({"A": modelo, "B": modelo}, X_test, y_test)

    assert list(tabela["modelo"]) == ["A", "B"]
    assert tabela["recall"].is_monotonic_decreasing


def test_save_metrics_grava_json_valido(tmp_path):
    destino = evaluation.save_metrics({"modelo": "teste", "recall": 0.93}, tmp_path / "m.json")

    assert json.loads(destino.read_text(encoding="utf-8"))["recall"] == 0.93

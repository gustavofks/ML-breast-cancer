"""Testes dos contratos das rotas."""

import pytest

pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main, servico  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.preprocessing import split_data  # noqa: E402


@pytest.fixture(scope="module")
def modelo():
    return servico.Servico.construir(servico.referencia_da_fase_1())


@pytest.fixture
def cliente(modelo, tmp_path):
    return TestClient(main.criar_app(modelo, diretorio_cache=tmp_path / "cache"))


@pytest.fixture(scope="module")
def caso():
    X, y = load_dataset()
    _, X_test, _, _ = split_data(X, y)
    return X_test.iloc[0].to_dict()


def test_health_confirma_que_o_modelo_carregou(cliente):
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok", "modelo_carregado": True}


def test_model_info_declara_origem_features_limiar_e_commit(cliente, monkeypatch):
    monkeypatch.setenv("APP_COMMIT", "abc1234")

    corpo = cliente.get("/model-info").json()

    assert corpo["familia"] == "Regressão Logística"
    assert len(corpo["features"]) == 30
    assert corpo["limiar"] == 0.5
    assert "Fase 1" in corpo["origem"]
    assert corpo["commit"] == "abc1234"


def test_model_info_sem_commit_declara_que_nao_sabe(cliente, monkeypatch):
    monkeypatch.delenv("APP_COMMIT", raising=False)

    assert cliente.get("/model-info").json()["commit"] == "desconhecido"


def test_predict_devolve_probabilidade_classe_e_limiar(cliente, caso, modelo):
    resposta = cliente.post("/predict", json={"medidas": caso})

    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["probabilidade"] == pytest.approx(modelo.prever(caso).probabilidade)
    assert corpo["limiar"] == 0.5
    assert corpo["classe"] in {"Benigno", "Maligno"}
    assert corpo["alertas"] == []


def test_predict_com_medida_faltante_responde_422_nomeando_a_medida(cliente, caso):
    incompleto = {k: v for k, v in caso.items() if k != "area_worst"}

    resposta = cliente.post("/predict", json={"medidas": incompleto})

    assert resposta.status_code == 422
    assert "area_worst" in resposta.text


def test_predict_com_medida_fora_da_distribuicao_alerta_e_ainda_responde(cliente, caso):
    """A sentinela registra; quem decide se confia no resultado e o medico."""
    deslocado = {**caso, "area_worst": 1e6}

    corpo = cliente.post("/predict", json={"medidas": deslocado}).json()

    assert [a["feature"] for a in corpo["alertas"]] == ["area_worst"]
    assert "probabilidade" in corpo

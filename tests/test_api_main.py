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


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from src.api import explicacao  # noqa: E402
from src.llm import narrator, prompts  # noqa: E402
from src.llm.provider import CachedProvider, FakeProvider  # noqa: E402

TEXTO_APROVAVEL = """## Leitura do caso
Texto de teste.

## Medidas que mais pesaram
Pesaram {citacao}.

## Ressalva
Esta leitura é apoio à decisão e não substitui a avaliação do médico responsável.
"""


def _semear_cache(modelo, caso, diretorio, texto):
    """Grava no cache o que `run_llm.py` gravaria para este caso."""
    pares = explicacao.contribuicoes(modelo, pd.DataFrame([caso]))
    previsao = modelo.prever(caso)
    contexto = narrator.contexto_do_paciente(
        probabilidade=previsao.probabilidade, limiar=previsao.limiar, contribuicoes=pares
    )
    citacao = ", ".join(c["feature"] for c in contexto["maiores_contribuicoes"])
    narrator.narrar(
        prompts.TEMPLATES["explicacao_paciente_v2"],
        contexto,
        CachedProvider(FakeProvider([texto.format(citacao=citacao)]), diretorio),
    )


def test_explain_traz_contribuicoes_ordenadas_por_peso(cliente, caso):
    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    pesos = [abs(c["contribuicao"]) for c in corpo["contribuicoes"]]
    assert pesos == sorted(pesos, reverse=True)
    assert len(corpo["contribuicoes"]) == 30
    assert {"probabilidade", "limiar", "classe", "alertas"} <= set(corpo)


def test_explain_sem_texto_no_cache_diz_por_que(cliente, caso):
    """Geracao e offline; caso que run_llm.py nao processou nao tem texto."""
    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    assert corpo["texto"] is None
    assert "run_llm.py" in corpo["texto_motivo"]


def test_explain_devolve_o_texto_gerado_offline(modelo, caso, tmp_path):
    """O cache gravado offline e encontrado pela mesma chave, online."""
    cache = tmp_path / "cache"
    _semear_cache(modelo, caso, cache, TEXTO_APROVAVEL)
    cliente = TestClient(main.criar_app(modelo, diretorio_cache=cache))

    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    assert corpo["texto"] is not None
    assert "não substitui" in corpo["texto"]
    assert corpo["texto_motivo"] is None


def test_texto_reprovado_nas_verificacoes_nao_chega_a_tela(modelo, caso, tmp_path):
    """Texto que falhou a verificacao de seguranca nao e mostrado a um medico."""
    cache = tmp_path / "cache"
    _semear_cache(modelo, caso, cache, "O diagnóstico é câncer confirmado.")
    cliente = TestClient(main.criar_app(modelo, diretorio_cache=cache))

    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    assert corpo["texto"] is None
    assert "seguranca" in corpo["texto_motivo"]


def test_explain_com_medida_faltante_responde_422(cliente, caso):
    incompleto = {k: v for k, v in caso.items() if k != "area_worst"}

    assert cliente.post("/explain", json={"medidas": incompleto}).status_code == 422

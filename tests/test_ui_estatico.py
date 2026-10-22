"""Testes dos casos do teste congelados a partir da API, para a vitrine estatica."""

import pytest

pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main, servico  # noqa: E402
from src.ui import app as painel  # noqa: E402
from src.ui import estatico  # noqa: E402


@pytest.fixture(scope="module")
def api():
    """A API de verdade, com o modelo servido e o cache de textos do projeto."""
    return TestClient(main.criar_app(servico.Servico.construir(servico.carregar_especificacao())))


@pytest.fixture(scope="module")
def dados(api):
    return estatico.dados_dos_casos(api)


def test_um_registro_por_caso_do_conjunto_de_teste(dados):
    assert len(dados["casos"]) == len(painel.casos_de_teste())


def test_numeros_vem_da_propria_api(api, dados):
    """A pagina estatica mostra o que a API serve, nao uma conta paralela."""
    medidas = painel.casos_de_teste().drop(columns="diagnostico_registrado").iloc[3].to_dict()
    resposta = api.post("/explain", json={"medidas": medidas}).json()

    caso = dados["casos"][3]
    assert caso["probabilidade"] == resposta["probabilidade"]
    assert caso["classe"] == resposta["classe"]
    assert caso["contribuicoes"] == resposta["contribuicoes"]
    assert caso["texto"] == resposta["texto"]


def test_registro_traz_o_diagnostico_registrado_e_o_modelo_servido(dados):
    assert dados["casos"][0]["diagnostico_registrado"] in {"Benigno", "Maligno"}
    assert dados["modelo"]["origem"]
    assert dados["modelo"]["limiar"]


def test_registro_traz_as_medidas_do_modelo_para_o_simulador(dados):
    for caso in dados["casos"]:
        assert list(caso["medidas"]) == dados["modelo"]["features"]

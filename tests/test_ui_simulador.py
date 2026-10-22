"""Testes do simulador, que refaz no navegador a conta do modelo servido."""

import pytest

pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main, servico  # noqa: E402
from src.genetic.chromosome import PADROES  # noqa: E402
from src.ui import app as painel  # noqa: E402
from src.ui import simulador  # noqa: E402


@pytest.fixture(scope="module")
def modelo():
    return servico.Servico.construir(servico.carregar_especificacao())


@pytest.fixture(scope="module")
def api(modelo, tmp_path_factory):
    return TestClient(
        main.criar_app(modelo, diretorio_cache=tmp_path_factory.mktemp("cache"))
    )


@pytest.fixture(scope="module")
def parametros(modelo):
    return simulador.parametros(modelo)


def test_parametros_trazem_as_medidas_do_modelo_na_ordem_dele(modelo, parametros):
    assert parametros["features"] == modelo.selecionadas
    for chave in ("media", "escala", "coeficientes", "media_do_fundo"):
        assert len(parametros[chave]) == len(modelo.selecionadas)
    assert parametros["limiar"] == modelo.especificacao.limiar


def test_conta_do_simulador_reproduz_a_api_em_todos_os_casos_do_teste(api, parametros):
    """A propriedade que autoriza o simulador: ele e a API, nao uma aproximacao dela."""
    casos = painel.casos_de_teste().drop(columns="diagnostico_registrado")

    for indice in range(len(casos)):
        medidas = casos.iloc[indice].to_dict()
        esperado = api.post("/explain", json={"medidas": medidas}).json()
        obtido = simulador.calcular(parametros, medidas)

        assert obtido["probabilidade"] == pytest.approx(esperado["probabilidade"], abs=1e-9)
        assert obtido["classe"] == esperado["classe"]
        assert [c["feature"] for c in obtido["contribuicoes"]] == [
            c["feature"] for c in esperado["contribuicoes"]
        ]
        for o, e in zip(obtido["contribuicoes"], esperado["contribuicoes"]):
            assert o["contribuicao"] == pytest.approx(e["contribuicao"], abs=1e-9)


def test_modelo_nao_linear_nao_tem_simulador():
    """Floresta e KNN nao cabem numa formula fechada; a vitrine diz isso na tela."""
    especificacao = servico.EspecificacaoDoModelo(
        familia="KNN",
        hiperparametros=dict(PADROES["KNN"]),
        features=("texture_mean", "area_se"),
        limiar=0.5,
        origem="teste",
    )

    assert simulador.parametros(servico.Servico.construir(especificacao)) is None


def test_parametros_sao_numeros_simples_que_viram_json(parametros):
    """Vao para um bloco JSON na pagina; numpy no meio quebraria a serializacao."""
    import json

    assert json.loads(json.dumps(parametros)) == parametros

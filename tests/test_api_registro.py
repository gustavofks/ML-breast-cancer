"""Testes do log estruturado."""

import io
import json

from src.api import registro


def _capturar(nome):
    fluxo = io.StringIO()
    return registro.configurar(nome=nome, fluxo=fluxo), fluxo


def test_cada_requisicao_vira_uma_linha_de_json_valido():
    logger, fluxo = _capturar("teste-linhas")

    registro.requisicao(logger, id="abc", rota="/predict", metodo="POST",
                        status=200, latencia_ms=12.5, resultado="Maligno")
    registro.requisicao(logger, id="def", rota="/health", metodo="GET",
                        status=200, latencia_ms=0.4, resultado=None)

    linhas = fluxo.getvalue().strip().splitlines()
    assert len(linhas) == 2
    primeira = json.loads(linhas[0])
    assert primeira["evento"] == "requisicao"
    assert {"id", "rota", "metodo", "status", "latencia_ms", "resultado", "momento"} <= set(
        primeira
    )


def test_registro_declara_que_nao_carrega_dado_identificavel():
    """Secao 6.2: a ausencia de dado identificavel e declarada, nao presumida."""
    logger, fluxo = _capturar("teste-identificavel")

    registro.requisicao(logger, id="abc", rota="/predict", metodo="POST",
                        status=200, latencia_ms=1.0, resultado="Benigno")
    registro.desvio(logger, id="abc", feature="area_worst", valor=1e6, desvios=42.0)

    for linha in fluxo.getvalue().strip().splitlines():
        assert json.loads(linha)["dado_identificavel"] is False


def test_desvio_sai_com_nivel_de_aviso():
    """Agregador filtra por nivel; alerta de desvio nao pode se perder entre requisicoes."""
    logger, fluxo = _capturar("teste-nivel")

    registro.desvio(logger, id="abc", feature="area_worst", valor=1e6, desvios=42.0)

    linha = json.loads(fluxo.getvalue())
    assert linha["evento"] == "desvio"
    assert linha["nivel"] == "WARNING"


def test_configurar_duas_vezes_nao_duplica_linhas():
    """Cada criar_app configura o logger; handler repetido dobraria cada linha."""
    fluxo = io.StringIO()
    registro.configurar(nome="teste-dup", fluxo=fluxo)
    logger = registro.configurar(nome="teste-dup", fluxo=fluxo)

    registro.requisicao(logger, id="x", rota="/health", metodo="GET",
                        status=200, latencia_ms=0.1, resultado=None)

    assert len(fluxo.getvalue().strip().splitlines()) == 1

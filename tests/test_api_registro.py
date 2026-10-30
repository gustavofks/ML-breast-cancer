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


def test_memoria_guarda_os_eventos_recentes_em_ordem_com_sequencia():
    """A vitrine mostra o log na tela; ela le os eventos desta memoria."""
    logger, _ = _capturar("teste-memoria")

    registro.requisicao(logger, id="a", rota="/health", metodo="GET",
                        status=200, latencia_ms=0.1, resultado=None)
    registro.desvio(logger, id="b", feature="area_se", valor=900.0, desvios=19.0)

    eventos = registro.memoria(logger).depois(0)
    assert [e["evento"] for e in eventos] == ["requisicao", "desvio"]
    assert [e["seq"] for e in eventos] == [1, 2]
    assert eventos[1]["nivel"] == "WARNING"


def test_memoria_devolve_so_o_que_veio_depois_da_sequencia_pedida():
    """A tela pede so o que e novo, e nao a memoria inteira a cada leitura."""
    logger, _ = _capturar("teste-depois")
    for i in range(3):
        registro.requisicao(logger, id=str(i), rota="/health", metodo="GET",
                            status=200, latencia_ms=0.1, resultado=None)

    assert [e["id"] for e in registro.memoria(logger).depois(2)] == ["2"]


def test_memoria_tem_capacidade_limitada():
    """Memoria sem limite cresceria com cada requisicao ate derrubar o processo."""
    fluxo = io.StringIO()
    logger = registro.configurar(nome="teste-capacidade", fluxo=fluxo, capacidade=3)
    for i in range(5):
        registro.requisicao(logger, id=str(i), rota="/health", metodo="GET",
                            status=200, latencia_ms=0.1, resultado=None)

    eventos = registro.memoria(logger).depois(0)
    assert [e["id"] for e in eventos] == ["2", "3", "4"]
    assert eventos[-1]["seq"] == 5


def test_memoria_e_a_linha_do_fluxo_trazem_os_mesmos_campos():
    """A tela e o agregador de log leem o mesmo evento, nao duas versoes dele."""
    logger, fluxo = _capturar("teste-mesmos-campos")

    registro.desvio(logger, id="b", feature="area_se", valor=900.0, desvios=19.0)

    linha = json.loads(fluxo.getvalue())
    evento = registro.memoria(logger).depois(0)[0]
    assert {k: v for k, v in evento.items() if k != "seq"} == linha

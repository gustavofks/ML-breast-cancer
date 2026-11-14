"""Testes do cliente do modelo de texto."""

import json

import httpx
import pytest

from src.assistente import modelo
from src.llm.provider import GeracaoIncompleta


def _servidor(resposta: dict, registro: list):
    def tratar(pedido: httpx.Request) -> httpx.Response:
        registro.append(json.loads(pedido.content))
        return httpx.Response(200, json=resposta)
    return httpx.Client(transport=httpx.MockTransport(tratar), base_url="http://falso")


def _resposta(texto, motivo="stop"):
    return {"choices": [{"message": {"content": texto}, "finish_reason": motivo}]}


def test_servidor_envia_mensagens_temperatura_e_semente_e_devolve_o_texto():
    registro = []
    cliente = modelo.ServidorCompativelOpenAI("http://falso", cliente=_servidor(_resposta("ok"), registro))

    texto = cliente.conversar([{"role": "user", "content": "oi"}], max_tokens=10,
                              temperatura=0.0, semente=7)

    assert texto == "ok"
    assert registro[0]["messages"] == [{"role": "user", "content": "oi"}]
    assert registro[0]["temperature"] == 0.0 and registro[0]["seed"] == 7
    assert registro[0]["max_tokens"] == 10


def test_resposta_cortada_pelo_limite_e_erro_e_nao_texto_pela_metade():
    cliente = modelo.ServidorCompativelOpenAI(
        "http://falso", cliente=_servidor(_resposta("meia frase", "length"), []))

    with pytest.raises(GeracaoIncompleta):
        cliente.conversar([{"role": "user", "content": "oi"}], max_tokens=3)


def test_modelo_falso_cumpre_o_protocolo_e_registra_as_chamadas():
    falso = modelo.ModeloFalso(["a", "b"])

    assert isinstance(falso, modelo.ModeloDeTexto)
    assert falso.conversar([], max_tokens=1) == "a"
    assert falso.conversar([], max_tokens=1) == "b"
    assert falso.conversar([], max_tokens=1) == "b"
    assert len(falso.chamadas) == 3


def test_modelo_falso_aceita_funcao_para_responder_conforme_o_pedido():
    falso = modelo.ModeloFalso(lambda mensagens: mensagens[-1]["content"].upper())

    assert falso.conversar([{"role": "user", "content": "eco"}], max_tokens=5) == "ECO"

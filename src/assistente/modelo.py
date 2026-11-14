"""Cliente do modelo de texto da Fase 3, atras de um Protocol.

O modelo professor e o modelo ajustado sao servidos pelo llama.cpp, que fala o
formato de chat da OpenAI. Um cliente so atende os dois, e o LangChain da
semana 4 conversa com o mesmo servidor. O ModeloFalso deixa a suite inteira
rodar sem GPU, pela mesma razao do FakeProvider da Fase 2.

Resposta cortada pelo limite de tokens e erro, e nao texto: a Fase 2 mostrou
que texto pela metade com aparencia plausivel e o defeito mais caro de achar.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import httpx

from src import config
from src.llm.provider import GeracaoIncompleta

Mensagens = list[dict[str, str]]


@runtime_checkable
class ModeloDeTexto(Protocol):
    def conversar(self, mensagens: Mensagens, *, max_tokens: int,
                  temperatura: float = 0.0, semente: int = config.SEED) -> str: ...


class ServidorCompativelOpenAI:
    def __init__(self, base_url: str, *, nome: str = "local", timeout: float = 900.0,
                 cliente: httpx.Client | None = None) -> None:
        self.nome = nome
        self._cliente = cliente or httpx.Client(base_url=base_url, timeout=timeout)

    def conversar(self, mensagens: Mensagens, *, max_tokens: int,
                  temperatura: float = 0.0, semente: int = config.SEED) -> str:
        resposta = self._cliente.post("/v1/chat/completions", json={
            "model": self.nome, "messages": mensagens, "max_tokens": max_tokens,
            "temperature": temperatura, "seed": semente,
        })
        resposta.raise_for_status()
        escolha = resposta.json()["choices"][0]
        if escolha.get("finish_reason") == "length":
            raise GeracaoIncompleta(f"Resposta cortada em {max_tokens} tokens.")
        return escolha["message"]["content"]


class ModeloFalso:
    """Respostas fixas, em ordem (a ultima se repete), ou calculadas por funcao."""

    def __init__(self, respostas: list[str] | Callable[[Mensagens], str]) -> None:
        self._respostas = respostas
        self.chamadas: list[Mensagens] = []

    def conversar(self, mensagens: Mensagens, *, max_tokens: int,
                  temperatura: float = 0.0, semente: int = config.SEED) -> str:
        self.chamadas.append(mensagens)
        if callable(self._respostas):
            return self._respostas(mensagens)
        return self._respostas[min(len(self.chamadas), len(self._respostas)) - 1]

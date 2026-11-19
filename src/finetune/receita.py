"""Receita do fine-tuning e conversao dos exemplos em prompt/completion.

O texto de treino e renderizado aqui, com o mesmo molde de chat e o mesmo
`enable_thinking=False` que a inferencia usa no llama.cpp. Se o TRL renderizasse
por conta propria, um detalhe do molde (o bloco de pensamento vazio do Qwen3.5,
por exemplo) poderia diferir entre treino e uso, e o modelo aprenderia a
responder a um texto que nunca vai ver.

A perda conta so a resposta: o prompt (trechos e pergunta) e entrada, e ensinar
o modelo a reproduzi-lo so gastaria capacidade.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from src import config

Renderizador = Callable[..., str]  # (mensagens, *, gerar: bool) -> str


@dataclass(frozen=True)
class Receita:
    modelo_base: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    taxa: float
    epocas: int
    acumulacao: int
    max_tokens: int
    semente: int


def receita_padrao() -> Receita:
    return Receita(
        modelo_base=config.FASE3_MODELO_BASE, lora_r=config.FASE3_LORA_R,
        lora_alpha=config.FASE3_LORA_ALPHA, lora_dropout=config.FASE3_LORA_DROPOUT,
        taxa=config.FASE3_TREINO_TAXA, epocas=config.FASE3_TREINO_EPOCAS,
        acumulacao=config.FASE3_TREINO_ACUMULACAO, max_tokens=config.FASE3_TREINO_MAX_TOKENS,
        semente=config.SEED,
    )


def para_prompt_completion(exemplo: dict, renderizar: Renderizador) -> dict[str, str]:
    """Divide a conversa renderizada no ponto em que a resposta comeca."""
    perguntas = [m for m in exemplo["mensagens"] if m["role"] != "assistant"]
    prompt = renderizar(perguntas, gerar=True)
    completo = renderizar(exemplo["mensagens"], gerar=False)
    if not completo.startswith(prompt):
        raise ValueError("O molde de chat nao prolonga o prompt: treino e inferencia divergiriam.")
    return {"prompt": prompt, "completion": completo[len(prompt):]}


def passos_por_epoca(n_exemplos: int, *, acumulacao: int) -> int:
    return math.ceil(n_exemplos / acumulacao)

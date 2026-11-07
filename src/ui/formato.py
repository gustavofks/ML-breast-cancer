"""Numeros como o publico da apresentacao le, e busca por nome nos artefatos."""

from __future__ import annotations

from typing import Any


def num(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor: float, casas: int = 1) -> str:
    return f"{num(valor * 100, casas)}%"


def por_nome(linhas: list[dict[str, Any]], chave: str, valor: Any) -> dict[str, Any]:
    return next(linha for linha in linhas if linha[chave] == valor)

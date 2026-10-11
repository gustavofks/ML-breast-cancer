"""Nucleo de dominio da API: qual modelo e servido, e como ele preve.

O modelo e **treinado na inicializacao**, a partir de uma especificacao, em vez
de carregado de um pickle. Pickle de scikit-learn so abre na mesma versao que o
gravou, e a base tem 455 amostras de treino: treinar custa menos de um segundo
e elimina a classe inteira de erro de versao. E o mesmo `build_pipeline` da
Fase 1, entao o modelo servido e o modelo avaliado sao o mesmo codigo.

A especificacao tem a forma do `melhor` que o algoritmo genetico grava. Qual
modelo servir e decisao tomada depois da avaliacao final dos bracos, e por isso
vive num arquivo, e nao aqui.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import config
from src.data import load_dataset
from src.genetic.chromosome import CONSTRUTORES, PADROES

ORIGEM_DA_REFERENCIA = "referência da Fase 1 (regressão logística, 30 medidas, limiar 0,5)"


@lru_cache(maxsize=1)
def colunas_do_dataset() -> tuple[str, ...]:
    """Nomes das 30 medidas, na ordem da base."""
    X, _ = load_dataset()
    return tuple(X.columns)


@dataclass(frozen=True)
class EspecificacaoDoModelo:
    """O que e servido: familia, hiperparametros, medidas e limiar."""

    familia: str
    hiperparametros: dict[str, Any] = field(hash=False)
    features: tuple[str, ...]
    limiar: float
    origem: str

    @classmethod
    def de_dicionario(cls, dados: dict[str, Any]) -> "EspecificacaoDoModelo":
        """Valida na carga.

        Especificacao errada tem que derrubar a inicializacao, e nao a primeira
        requisicao de um medico.
        """
        familia = dados["familia"]
        if familia not in CONSTRUTORES:
            raise ValueError(f"familia desconhecida: {familia!r}")

        features = tuple(dados["features"])
        if not features:
            raise ValueError("o modelo servido precisa de ao menos uma medida")
        desconhecidas = sorted(set(features) - set(colunas_do_dataset()))
        if desconhecidas:
            raise ValueError(f"medidas fora da base: {desconhecidas}")

        limiar = float(dados["limiar"])
        if not 0.0 < limiar < 1.0:
            raise ValueError(f"limiar fora de (0, 1): {limiar}")

        return cls(
            familia=familia,
            hiperparametros=dict(dados["hiperparametros"]),
            features=features,
            limiar=limiar,
            origem=str(dados.get("origem", "não declarada")),
        )

    def para_dicionario(self) -> dict[str, Any]:
        return {
            "familia": self.familia,
            "hiperparametros": dict(self.hiperparametros),
            "features": list(self.features),
            "limiar": self.limiar,
            "origem": self.origem,
        }


def referencia_da_fase_1() -> EspecificacaoDoModelo:
    """O modelo da Fase 1: regressao logistica padrao, 30 medidas, limiar 0,5."""
    return EspecificacaoDoModelo(
        familia="Regressão Logística",
        hiperparametros=dict(PADROES["Regressão Logística"]),
        features=colunas_do_dataset(),
        limiar=0.5,
        origem=ORIGEM_DA_REFERENCIA,
    )


def carregar_especificacao(
    caminho: Path = config.MODELO_SERVIDO_FILE,
) -> EspecificacaoDoModelo:
    """Le a especificacao do arquivo, ou devolve a referencia declarada como tal."""
    if not caminho.exists():
        return referencia_da_fase_1()
    return EspecificacaoDoModelo.de_dicionario(json.loads(caminho.read_text(encoding="utf-8")))

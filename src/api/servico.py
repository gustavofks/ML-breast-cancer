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
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.data import load_dataset
from src.genetic.chromosome import CONSTRUTORES, PADROES
from src.preprocessing import build_pipeline, split_data

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


class MedidasInvalidas(ValueError):
    """Entrada que nao pode virar previsao. A mensagem nomeia cada medida."""


@dataclass(frozen=True)
class Previsao:
    probabilidade: float
    limiar: float
    classe: str


@dataclass
class Servico:
    """Modelo treinado uma vez, com o que a explicacao e a sentinela precisam.

    `X_train` guarda as 30 medidas, e nao so as do modelo: a sentinela de desvio
    olha todas, e o explicador recorta as que precisa.
    """

    especificacao: EspecificacaoDoModelo
    pipeline: Pipeline
    X_train: pd.DataFrame

    @classmethod
    def construir(cls, especificacao: EspecificacaoDoModelo) -> "Servico":
        X, y = load_dataset()
        # O teste e descartado: o modelo servido nunca o viu, como o da avaliacao.
        X_train, _, y_train, _ = split_data(X, y)
        selecionadas = [c for c in X.columns if c in especificacao.features]

        estimador = CONSTRUTORES[especificacao.familia](dict(especificacao.hiperparametros))
        pipeline = build_pipeline(estimador).fit(X_train[selecionadas], y_train)
        return cls(especificacao, pipeline, X_train)

    @property
    def selecionadas(self) -> list[str]:
        """Medidas do modelo na ordem da base, e nao na do arquivo.

        E a ordem que `decodificar` usa no algoritmo genetico, e portanto a que o
        modelo avaliado viu.
        """
        return [c for c in self.X_train.columns if c in self.especificacao.features]

    def validar(self, medidas: dict[str, float]) -> pd.DataFrame:
        """Uma linha com as medidas do modelo, ou `MedidasInvalidas` nomeando o defeito.

        Medida desconhecida e recusada, e nao ignorada: um nome digitado errado
        passaria em silencio, e a previsao sairia sem a medida que o usuario
        acha que mandou.
        """
        desconhecidas = sorted(set(medidas) - set(self.X_train.columns))
        if desconhecidas:
            raise MedidasInvalidas(f"medidas desconhecidas: {desconhecidas}")
        faltantes = [c for c in self.selecionadas if c not in medidas]
        if faltantes:
            raise MedidasInvalidas(f"medidas faltantes: {faltantes}")
        nao_finitas = sorted(k for k, v in medidas.items() if not math.isfinite(float(v)))
        if nao_finitas:
            raise MedidasInvalidas(f"medidas nao finitas: {nao_finitas}")
        return pd.DataFrame([{c: float(medidas[c]) for c in self.selecionadas}])

    def prever(self, medidas: dict[str, float]) -> Previsao:
        """Probabilidade de maligno e a classe que o limiar servido decide."""
        linha = self.validar(medidas)
        probabilidade = float(self.pipeline.predict_proba(linha)[0, config.POSITIVE_LABEL])
        limiar = self.especificacao.limiar
        classe = config.CLASS_NAMES[1] if probabilidade >= limiar else config.CLASS_NAMES[0]
        return Previsao(probabilidade=probabilidade, limiar=limiar, classe=classe)

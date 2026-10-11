"""API da Fase 2: previsao, explicacao, metricas e informacao do modelo.

Construida por fabrica (`criar_app`), e nao no import do modulo: o modelo e
treinado dentro dela, e os testes injetam um servico ja treinado em vez de
treinar um por arquivo. O Uvicorn chama a fabrica com `--factory`.

A API nao tem estado alem dos contadores em memoria: o modelo e carregado uma
vez, na inicializacao, e qualquer replica responde igual. E o que a secao 6.4 do
design chama de servico preparado para escalar.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.api import monitoring, registro
from src.api import servico as dominio


class Medidas(BaseModel):
    """Corpo de /predict e /explain: medidas do nucleo celular, por nome."""

    medidas: dict[str, float]


def criar_app(
    servico: dominio.Servico | None = None,
    *,
    diretorio_cache: Path = config.LLM_CACHE_DIR,
) -> FastAPI:
    """Monta a API em torno de um modelo ja treinado, ou treina o especificado."""
    modelo = servico or dominio.Servico.construir(dominio.carregar_especificacao())
    sentinela = monitoring.Sentinela.de_treino(modelo.X_train, k=config.API_LIMIAR_DE_DESVIO)
    monitor = monitoring.Monitor()
    logger = registro.configurar()

    app = FastAPI(title="Apoio ao diagnóstico de câncer de mama", version="2.0")
    app.state.diretorio_cache = Path(diretorio_cache)

    def prever(medidas: dict[str, float]):
        """Previsao com validacao e sentinela; comum a /predict e /explain."""
        try:
            previsao = modelo.prever(medidas)
        except dominio.MedidasInvalidas as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro

        alertas = sentinela.verificar(medidas)
        monitor.registrar_alertas(alertas)
        for alerta in alertas:
            registro.desvio(
                logger, feature=alerta.feature, valor=alerta.valor, desvios=alerta.desvios
            )
        return previsao, alertas

    @app.get("/health")
    def health():
        return {"status": "ok", "modelo_carregado": True}

    @app.get("/model-info")
    def model_info():
        # O commit e lido a cada chamada, e nao na fabrica: o ambiente e a fonte
        # de verdade, e e o que o container preenche no build.
        return {
            **modelo.especificacao.para_dicionario(),
            "features": modelo.selecionadas,
            "commit": os.environ.get(config.API_VARIAVEL_DE_COMMIT, "desconhecido"),
        }

    @app.post("/predict")
    def predict(corpo: Medidas):
        previsao, alertas = prever(corpo.medidas)
        return {
            "probabilidade": previsao.probabilidade,
            "limiar": previsao.limiar,
            "classe": previsao.classe,
            "alertas": [vars(alerta) for alerta in alertas],
        }

    return app

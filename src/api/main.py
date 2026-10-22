"""API da Fase 2: previsao, explicacao, metricas e informacao do modelo.

Construida por fabrica (`criar_app`), e nao no import do modulo: o modelo e
treinado dentro dela, e os testes injetam um servico ja treinado em vez de
treinar um por arquivo. O Uvicorn chama a fabrica com `--factory`.

A API nao tem estado alem dos contadores em memoria: o modelo e carregado uma
vez, na inicializacao, e qualquer replica responde igual. E o que a secao 6.4 do
design chama de servico preparado para escalar.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from src import config
from src.api import explicacao, monitoring, registro
from src.api import servico as dominio
from src.llm import narrator, prompts
from src.llm.provider import CacheVazioError, CachedProvider

# Template das explicacoes por paciente. E o mesmo que `scripts/run_llm.py` usa
# no lote: a chave do cache inclui o prompt, entao outro template nunca acharia
# o texto gravado.
TEMPLATE_DO_PACIENTE = "explicacao_paciente_v2"

# Rotas que nao geram log nem metrica. Ler o log pela tela geraria uma linha
# nova a cada leitura, e a tela passaria a mostrar o proprio eco.
ROTAS_SILENCIOSAS = frozenset({"/logs/recentes"})


class Medidas(BaseModel):
    """Corpo de /predict e /explain: medidas do nucleo celular, por nome."""

    medidas: dict[str, float]


def criar_app(
    servico: dominio.Servico | None = None,
    *,
    diretorio_cache: Path = config.LLM_CACHE_DIR,
    logger: logging.Logger | None = None,
) -> FastAPI:
    """Monta a API em torno de um modelo ja treinado, ou treina o especificado.

    `logger` existe para quem monta uma segunda API no mesmo processo, como a
    vitrine: com o logger padrao, as duas escreveriam no mesmo log e na mesma
    memoria de eventos, e as chamadas da montagem apareceriam no monitoramento
    da API servida.
    """
    modelo = servico or dominio.Servico.construir(dominio.carregar_especificacao())
    sentinela = monitoring.Sentinela.de_treino(modelo.X_train, k=config.API_LIMIAR_DE_DESVIO)
    monitor = monitoring.Monitor()
    logger = logger or registro.configurar()
    eventos = registro.memoria(logger)

    app = FastAPI(title="Apoio ao diagnóstico de câncer de mama", version="2.0")
    app.state.diretorio_cache = Path(diretorio_cache)

    @app.middleware("http")
    async def medir(request: Request, chamar):
        """Uma linha de log e uma amostra de metrica por requisicao, com um id."""
        identificador = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.id = identificador
        inicio = time.perf_counter()
        resposta = await chamar(request)
        segundos = time.perf_counter() - inicio

        # Rotulo pelo molde da rota, nunca pelo caminho bruto: uma varredura de
        # URLs criaria um rotulo por tentativa e explodiria a cardinalidade. Uma
        # aplicacao montada (o painel) sobrescreve a rota do escopo com a dela;
        # o prefixo da montagem, em `root_path`, evita que suas paginas sejam
        # contadas como a raiz.
        rota = request.scope.get("route")
        if rota is None:
            nome = "nao_encontrada"
        else:
            nome = request.scope.get("root_path", "") + rota.path
        if nome in ROTAS_SILENCIOSAS:
            return resposta

        monitor.registrar(nome, resposta.status_code, segundos)
        registro.requisicao(
            logger,
            id=identificador,
            rota=nome,
            metodo=request.method,
            status=resposta.status_code,
            latencia_ms=round(segundos * 1000, 2),
            resultado=getattr(request.state, "resultado", None),
        )
        resposta.headers["X-Request-ID"] = identificador
        return resposta

    def prever(request: Request, medidas: dict[str, float]):
        """Previsao com validacao e sentinela; comum a /predict e /explain."""
        try:
            previsao = modelo.prever(medidas)
        except dominio.MedidasInvalidas as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro

        alertas = sentinela.verificar(medidas)
        monitor.registrar_alertas(alertas)
        for alerta in alertas:
            registro.desvio(
                logger,
                id=request.state.id,
                feature=alerta.feature,
                valor=alerta.valor,
                desvios=alerta.desvios,
            )
        request.state.resultado = previsao.classe
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

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return monitor.exportar()

    @app.get("/logs/recentes")
    def logs_recentes(depois: int = 0):
        """Eventos de log em memoria com sequencia maior que `depois`.

        O mesmo JSON das linhas do stdout. Existe para a vitrine mostrar o log
        na tela; um agregador de verdade le o stdout.
        """
        return {"eventos": eventos.depois(depois)}

    @app.post("/predict")
    def predict(corpo: Medidas, request: Request):
        previsao, alertas = prever(request, corpo.medidas)
        return {
            "probabilidade": previsao.probabilidade,
            "limiar": previsao.limiar,
            "classe": previsao.classe,
            "alertas": [vars(alerta) for alerta in alertas],
        }

    @app.post("/explain")
    def explain(corpo: Medidas, request: Request):
        previsao, alertas = prever(request, corpo.medidas)
        pares = explicacao.contribuicoes(modelo, pd.DataFrame([corpo.medidas]))
        texto, motivo = texto_do_cache(previsao, pares)
        return {
            "probabilidade": previsao.probabilidade,
            "limiar": previsao.limiar,
            "classe": previsao.classe,
            "alertas": [vars(alerta) for alerta in alertas],
            "contribuicoes": [
                {"feature": nome, "contribuicao": valor}
                for nome, valor in sorted(pares, key=lambda par: abs(par[1]), reverse=True)
            ],
            "texto": texto,
            "texto_motivo": motivo,
        }

    def texto_do_cache(previsao, pares) -> tuple[str | None, str | None]:
        """Texto gerado offline para este caso, se existir e se passou nas verificacoes.

        O cache e consultado pela mesma chave que `run_llm.py` gravou: mesmo
        contexto, mesmo template, mesmo modelo de linguagem. Sem provedor por
        tras, nada e gerado aqui — a geracao ao vivo esta fora do escopo, porque a
        camada gratuita da nuvem nao comporta modelo de linguagem.

        Texto reprovado nao e devolvido. Uma frase que falhou a verificacao de
        seguranca nao chega a tela de um medico; o motivo diz qual falhou.
        """
        contexto = narrator.contexto_do_paciente(
            probabilidade=previsao.probabilidade, limiar=previsao.limiar, contribuicoes=pares
        )
        try:
            narrativa = narrator.narrar(
                prompts.TEMPLATES[TEMPLATE_DO_PACIENTE],
                contexto,
                CachedProvider(None, app.state.diretorio_cache),
            )
        except CacheVazioError:
            return None, "texto gerado apenas para os casos processados por scripts/run_llm.py"

        if not narrativa.qualidade.aprovado:
            reprovadas = [
                nome for nome, v in narrativa.qualidade.verificacoes.items() if not v.aprovado
            ]
            return None, f"texto reprovado nas verificações: {', '.join(reprovadas)}"
        return narrativa.texto, None

    return app

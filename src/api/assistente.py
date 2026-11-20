"""Rotas da Fase 3: pergunta livre, fluxo com aprovacao do medico e auditoria.

O servico do assistente e montado uma vez e injetado em criar_app; sem ele, as
rotas nao existem e a API da Fase 2 fica como era. Sem o servidor do modelo, a
resposta e 503 com o motivo: o assistente nunca inventa resposta no lugar do
modelo (spec, secao 10).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.assistente import cadeia, ferramentas, fluxo
from src.assistente.auditoria import Auditoria
from src.assistente.instrucoes import INSTRUCOES
from src.clinica import banco
from src.corpus import busca


def _erros_de_conexao() -> tuple[type[BaseException], ...]:
    erros: list[type[BaseException]] = [ConnectionError, TimeoutError, httpx.HTTPError]
    try:
        import openai

        erros += [openai.APIConnectionError, openai.APITimeoutError]
    except ImportError:
        pass
    return tuple(erros)


INDISPONIVEL = _erros_de_conexao()
MENSAGEM_INDISPONIVEL = "modelo indisponível: o servidor do assistente não respondeu"


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=1, max_length=2000)
    paciente: str | None = None


class Inicio(BaseModel):
    paciente: str


class Decisao(BaseModel):
    acao: Literal["aprovar", "editar", "rejeitar"]
    medico: str = Field(min_length=1)
    texto: str | None = None


@dataclass
class ServicoAssistente:
    con: sqlite3.Connection
    assistente: cadeia.Assistente
    fluxo: fluxo.Fluxo
    auditoria: Auditoria

    @classmethod
    def construir(cls, classificar) -> "ServicoAssistente":
        """Banco reconstruido dos prontuarios anonimizados se faltar, modelo ajustado
        pelo llama.cpp e pausas do fluxo em SQLite."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        if not config.CLINICA_BANCO_FILE.exists():
            registros = [json.loads(linha) for linha in
                         config.CLINICA_PRONTUARIOS_FILE.read_text(encoding="utf-8").splitlines()]
            banco.criar(config.CLINICA_BANCO_FILE, registros)
        con = banco.conectar(config.CLINICA_BANCO_FILE)
        modelo = cadeia.modelo_local(config.FASE3_AJUSTADO_URL, nome="ajustado")
        assistente = cadeia.Assistente(
            modelo, busca.indice_padrao(),
            instrucoes=INSTRUCOES if config.FASE3_USAR_INSTRUCOES else None)
        auditoria = Auditoria(config.FASE3_AUDITORIA_FILE)
        config.FASE3_EXECUCAO_DIR.mkdir(parents=True, exist_ok=True)
        pausa = sqlite3.connect(config.FASE3_CHECKPOINTS_FILE, check_same_thread=False)
        treino = config.FASE3_TREINO_DIR / "treino.json"
        versao = {"modelo": config.FASE3_MODELO_BASE, "adaptador_sha256": (
            json.loads(treino.read_text(encoding="utf-8")).get("adaptador_sha256")
            if treino.exists() else None)}
        dep = fluxo.Dependencias(con=con, assistente=assistente, auditoria=auditoria,
                                 classificar=classificar, versao=versao)
        return cls(con=con, assistente=assistente, fluxo=fluxo.Fluxo(dep, SqliteSaver(pausa)),
                   auditoria=auditoria)


def rotas(s: ServicoAssistente) -> APIRouter:
    r = APIRouter()

    @r.post("/assistente/perguntar")
    def perguntar(corpo: Pergunta):
        try:
            contexto = (ferramentas.contexto_da_paciente(s.con, corpo.paciente)
                        if corpo.paciente else None)
        except LookupError as erro:
            raise HTTPException(status_code=404, detail=str(erro)) from erro
        try:
            resposta = s.assistente.responder(corpo.pergunta, contexto)
        except INDISPONIVEL as erro:
            raise HTTPException(status_code=503, detail=MENSAGEM_INDISPONIVEL) from erro
        execucao = uuid.uuid4().hex
        campos = {"paciente": corpo.paciente} if corpo.paciente else {}
        s.auditoria.registrar(execucao, "pergunta_livre", modo=resposta.modo,
                              trechos=[t.id for t in resposta.trechos],
                              aprovado=resposta.verificacao.aprovado,
                              motivos=list(resposta.verificacao.motivos), **campos)
        return {"execucao": execucao, "resposta": resposta.texto_final, "modo": resposta.modo,
                "aprovado": resposta.verificacao.aprovado,
                "motivos": list(resposta.verificacao.motivos),
                "trechos": [{"id": t.id, "texto": t.texto} for t in resposta.trechos]}

    @r.post("/fluxo")
    def iniciar(corpo: Inicio):
        try:
            return s.fluxo.iniciar(corpo.paciente)
        except LookupError as erro:
            raise HTTPException(status_code=404, detail=str(erro)) from erro
        except INDISPONIVEL as erro:
            raise HTTPException(status_code=503, detail=MENSAGEM_INDISPONIVEL) from erro

    @r.post("/fluxo/{execucao}/decisao")
    def decidir(execucao: str, corpo: Decisao):
        try:
            return s.fluxo.decidir(execucao, corpo.model_dump())
        except LookupError as erro:
            raise HTTPException(status_code=404, detail=str(erro)) from erro
        except fluxo.EstadoInvalido as erro:
            raise HTTPException(status_code=409, detail=str(erro)) from erro
        except ValueError as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro

    @r.get("/auditoria/{execucao}")
    def auditoria(execucao: str):
        eventos = s.auditoria.ler(execucao)
        if not eventos:
            raise HTTPException(status_code=404, detail=f"Execucao sem registro: {execucao}")
        return {"execucao": execucao, "eventos": eventos}

    return r

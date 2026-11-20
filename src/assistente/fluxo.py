"""Fluxo LangGraph do assistente: do prontuario a sugestao aprovada pelo medico.

Um no por passo da secao 7 do spec, cada um com seu evento de auditoria. A
sugestao so chega ao medico se passou nas guardas; reprovada, vira alerta e o
fluxo vai direto a registrar. O no aprovacao_medica para o grafo com interrupt
ate o medico decidir, e o estado fica no SQLite: a pausa sobrevive a reinicio.
O LangGraph executa de novo o no interrompido ao retomar, entao nada nele tem
efeito antes do interrupt.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src import config
from src.assistente import ferramentas, guardas
from src.assistente.auditoria import Auditoria
from src.assistente.cadeia import Assistente
from src.clinica import banco
from src.corpus.protocolos import Trecho

ACOES = ("aprovar", "editar", "rejeitar")
# Prazo do protocolo vigente (PROT-13 v2), o mesmo que esta no indice da busca.
PRAZO_DIAS = config.CLINICA_PRAZO_PENDENCIA_DIAS_V2


class EstadoInvalido(RuntimeError):
    """Decisao pedida para uma execucao que nao esta esperando o medico."""


class Estado(TypedDict, total=False):
    execucao: str
    paciente: str
    contexto: str
    pendencias: list[dict]
    classificacao: dict | None
    pergunta: str
    trechos: list[dict]
    sugestao: str
    cortada: bool
    verificacao: dict
    alertas: list[str]
    decisao: dict | None


@dataclass
class Dependencias:
    con: sqlite3.Connection
    assistente: Assistente
    auditoria: Auditoria
    classificar: Callable[[dict], object] | None  # Servico.prever da Fase 2
    versao: dict  # modelo e adaptador servidos, para a auditoria


def pergunta_do_caso(classificacao: dict | None, pendencias: list[dict]) -> str:
    """Pergunta que guia a busca e o modelo, montada a partir do caso."""
    partes = []
    if classificacao:
        partes.append("Qual a conduta depois de uma PAAF com triagem "
                      f"{classificacao['classe'].lower()} pelo classificador?")
    if pendencias:
        partes.append("Qual a conduta para exame solicitado e sem resultado além do prazo, "
                      "na gestão de pendências de exames?")
    return " ".join(partes) or "Qual a conduta de seguimento para esta paciente?"


def validar_decisao(decisao: dict) -> dict:
    acao = decisao.get("acao")
    if acao not in ACOES:
        raise ValueError(f"acao invalida: {acao!r}; use uma de {ACOES}")
    if not str(decisao.get("medico") or "").strip():
        raise ValueError("a decisao precisa do identificador do medico")
    if acao == "editar" and not str(decisao.get("texto") or "").strip():
        raise ValueError("editar exige o texto editado")
    return {"acao": acao, "medico": decisao["medico"], "texto": decisao.get("texto")}


def _grafo(dep: Dependencias) -> StateGraph:
    def auditar(e: Estado, no: str, **campos) -> None:
        dep.auditoria.registrar(e["execucao"], no, **campos)

    def receber(e: Estado) -> dict:
        if banco.dados_paciente(dep.con, e["paciente"]) is None:
            raise LookupError(f"Paciente inexistente: {e['paciente']}")
        auditar(e, "receber", paciente=e["paciente"], **dep.versao)
        return {"alertas": [], "decisao": None}

    def consultar_prontuario(e: Estado) -> dict:
        contexto = ferramentas.contexto_da_paciente(dep.con, e["paciente"], prazo_dias=PRAZO_DIAS)
        auditar(e, "consultar_prontuario", exames=len(banco.exames(dep.con, e["paciente"])))
        return {"contexto": contexto}

    def verificar_pendencias(e: Estado) -> dict:
        pendencias = [{"exame": p["exame"], "solicitado_em": p["solicitado_em"]}
                      for p in banco.pendencias(dep.con, e["paciente"], hoje=config.CLINICA_HOJE,
                                                prazo_dias=PRAZO_DIAS)]
        auditar(e, "verificar_pendencias", pendencias=len(pendencias))
        return {"pendencias": pendencias}

    def classificar(e: Estado) -> dict:
        paafs = [x for x in banco.exames(dep.con, e["paciente"]) if x["exame"] == "PAAF" and x["medidas"]]
        if not paafs or dep.classificar is None:
            auditar(e, "classificar", classificado=False)
            return {"classificacao": None}
        previsao = dep.classificar(paafs[-1]["medidas"])
        c = {"probabilidade": round(previsao.probabilidade, 4), "limiar": previsao.limiar,
             "classe": previsao.classe, "distancia": round(previsao.probabilidade - previsao.limiar, 4)}
        auditar(e, "classificar", classificado=True, **c)
        return {"classificacao": c}

    def buscar_protocolos(e: Estado) -> dict:
        pergunta = pergunta_do_caso(e["classificacao"], e["pendencias"])
        trechos = dep.assistente.recuperar(pergunta)
        auditar(e, "buscar_protocolos", trechos=[t.id for t in trechos])
        return {"pergunta": pergunta, "trechos": [vars(t) for t in trechos]}

    def sugerir(e: Estado) -> dict:
        trechos = [Trecho(**t) for t in e["trechos"]]
        saida = dep.assistente.redigir(e["pergunta"], trechos, e["contexto"])
        auditar(e, "sugerir", caracteres=len(saida["texto"]), cortada=saida["cortada"])
        return {"sugestao": saida["texto"], "cortada": saida["cortada"]}

    def verificar(e: Estado) -> dict:
        v = guardas.verificar(e["sugestao"], [Trecho(**t) for t in e["trechos"]], cortada=e["cortada"])
        auditar(e, "verificar", aprovado=v.aprovado, motivos=list(v.motivos))
        return {"verificacao": {"aprovado": v.aprovado, "motivos": list(v.motivos)}}

    def alertar(e: Estado) -> dict:
        alertas = [f"Pendência vencida: {p['exame']} solicitado em {p['solicitado_em']}"
                   for p in e["pendencias"]]
        c = e.get("classificacao")
        if c and abs(c["distancia"]) < config.FASE3_MARGEM_LIMIAR:
            alertas.append(f"Caso perto do limiar do classificador (probabilidade "
                           f"{c['probabilidade']:.2f}, limiar {c['limiar']:.2f})")
        if not e["verificacao"]["aprovado"]:
            alertas.append("Sugestão retida pelas verificações: " + "; ".join(e["verificacao"]["motivos"]))
        auditar(e, "alertar", alertas=alertas)
        return {"alertas": alertas}

    def aprovacao_medica(e: Estado) -> dict:
        decisao = validar_decisao(interrupt({
            "sugestao": e["sugestao"], "trechos": [t["id"] for t in e["trechos"]],
            "alertas": e["alertas"], "classificacao": e.get("classificacao")}))
        auditar(e, "aprovacao_medica", acao=decisao["acao"], medico=decisao["medico"],
                editada=decisao["acao"] == "editar")
        return {"decisao": decisao}

    def registrar(e: Estado) -> dict:
        d = e.get("decisao")
        auditar(e, "registrar", desfecho=d["acao"] if d else "sugestao_retida",
                alertas=len(e["alertas"]))
        return {}

    g = StateGraph(Estado)
    passos = [receber, consultar_prontuario, verificar_pendencias, classificar, buscar_protocolos,
              sugerir, verificar, alertar, aprovacao_medica, registrar]
    for passo in passos:
        g.add_node(passo.__name__, passo)
    g.add_edge(START, "receber")
    for antes, depois in zip(passos[:7], passos[1:8]):
        g.add_edge(antes.__name__, depois.__name__)
    g.add_conditional_edges("alertar", lambda e: "aprovacao_medica" if e["verificacao"]["aprovado"]
                            else "registrar", ["aprovacao_medica", "registrar"])
    g.add_edge("aprovacao_medica", "registrar")
    g.add_edge("registrar", END)
    return g


class Fluxo:
    CAMPOS = ("paciente", "classificacao", "pergunta", "trechos", "sugestao", "verificacao",
              "alertas", "decisao")

    def __init__(self, dep: Dependencias, checkpointer) -> None:
        self._grafo = _grafo(dep).compile(checkpointer=checkpointer)

    @staticmethod
    def _cfg(execucao: str) -> dict:
        return {"configurable": {"thread_id": execucao}}

    def iniciar(self, paciente: str, execucao: str | None = None) -> dict:
        execucao = execucao or uuid.uuid4().hex
        self._grafo.invoke({"execucao": execucao, "paciente": paciente}, self._cfg(execucao))
        return self.situacao(execucao)

    def situacao(self, execucao: str) -> dict:
        estado = self._grafo.get_state(self._cfg(execucao))
        if not estado.values:
            raise LookupError(f"Execucao inexistente: {execucao}")
        if "aprovacao_medica" in estado.next:
            status = "aguardando_medico"
        else:
            status = "em_andamento" if estado.next else "concluido"
        return {"execucao": execucao, "status": status,
                **{c: estado.values.get(c) for c in self.CAMPOS}}

    def decidir(self, execucao: str, decisao: dict) -> dict:
        if self.situacao(execucao)["status"] != "aguardando_medico":
            raise EstadoInvalido(f"Execucao {execucao} nao esta esperando o medico")
        validar_decisao(decisao)  # antes de retomar: decisao invalida nao toca o grafo
        self._grafo.invoke(Command(resume=decisao), self._cfg(execucao))
        return self.situacao(execucao)

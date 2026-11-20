"""Ferramentas de prontuario: as consultas fixas do banco, presas a uma paciente.

A paciente e escolhida pelo codigo (o medico seleciona na tela), e nenhuma
ferramenta recebe argumento: nao ha como o modelo, ou um texto injetado no
prontuario, pedir outra paciente. As ferramentas sao StructuredTool do
LangChain, chamadas pela cadeia e pelo grafo da semana 5; o modelo nao escolhe
ferramenta, porque o ajustado nunca viu chamada de ferramenta no treino.
"""

from __future__ import annotations

import sqlite3

from langchain_core.tools import StructuredTool

from src import config
from src.clinica import banco


def _sem_origem(registro: dict | None) -> dict | None:
    """A origem e metadado do gerador, nao dado clinico."""
    return None if registro is None else {k: v for k, v in registro.items() if k != "origem"}


def ferramentas_da_paciente(con: sqlite3.Connection, paciente: str, *,
                            hoje: str = config.CLINICA_HOJE,
                            prazo_dias: int = config.CLINICA_PRAZO_PENDENCIA_DIAS) -> list[StructuredTool]:
    consultas = {
        "dados_paciente": ("Idade e historico da paciente selecionada.",
                           lambda: _sem_origem(banco.dados_paciente(con, paciente))),
        "exames": ("Exames da paciente selecionada, com data e resultado.",
                   lambda: [_sem_origem(e) for e in banco.exames(con, paciente)]),
        "evolucoes": ("Evolucoes clinicas da paciente selecionada.",
                      lambda: [_sem_origem(e) for e in banco.evolucoes(con, paciente)]),
        "pendencias": ("Exames solicitados sem resultado alem do prazo.",
                       lambda: [_sem_origem(e) for e in banco.pendencias(
                           con, paciente, hoje=hoje, prazo_dias=prazo_dias)]),
    }
    return [StructuredTool.from_function(func=f, name=nome, description=descricao)
            for nome, (descricao, f) in consultas.items()]


def contexto_da_paciente(con: sqlite3.Connection, paciente: str, **prazo) -> str:
    """Resumo do prontuario para a mensagem do modelo. As medidas da PAAF ficam
    de fora: quem as le e o classificador da Fase 2, no fluxo da semana 5."""
    saidas = {f.name: f.invoke({}) for f in ferramentas_da_paciente(con, paciente, **prazo)}
    dados = saidas["dados_paciente"]
    if dados is None:
        raise LookupError(f"Paciente inexistente: {paciente}")
    linhas = [f"Paciente: {paciente}, {dados['idade']} anos.", f"Histórico: {dados['historico']}"]
    linhas.append("Exames:")
    for e in saidas["exames"]:
        resultado = e["resultado"] or "sem resultado"
        linhas.append(f"- {e['solicitado_em']} {e['exame']}: {resultado}")
    if saidas["evolucoes"]:
        linhas.append("Evoluções:")
        linhas += [f"- {e['data']}: {e['texto']}" for e in saidas["evolucoes"]]
    if saidas["pendencias"]:
        linhas.append("Pendências vencidas:")
        linhas += [f"- {e['exame']} solicitado em {e['solicitado_em']}" for e in saidas["pendencias"]]
    return "\n".join(linhas)

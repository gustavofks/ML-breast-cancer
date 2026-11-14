"""Anonimizacao dos prontuarios da clinica ficticia.

Duas camadas, porque cada uma cobre o que a outra nao ve. A primeira troca o que
o cadastro conhece (nome, CPF, telefone, e-mail, endereco, nascimento de cada
paciente) por pseudonimo ou marcador. A segunda varre o texto livre com padroes
de CPF, telefone e e-mail, e pega identificador de quem nao esta no cadastro,
como um familiar citado na evolucao.

Datas clinicas ficam, porque pendencia e calculada contra o "hoje" unico da
clinica; a data de nascimento sai e vira idade (decisao registrada no plano da
semana 1). O mapa de pseudonimos volta separado dos registros e e gravado fora
do Git.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from src import config
from src.clinica.gerador import Clinica, Paciente

_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_TELEFONE = re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


@dataclass(frozen=True)
class Resultado:
    registros: list[dict]
    mapa: dict[str, str]  # id interno -> pseudonimo; nunca vai para os registros


def limpar_texto(texto: str) -> str:
    """Segunda camada: padroes, sem depender do cadastro."""
    texto = _EMAIL.sub("[EMAIL]", texto)
    texto = _CPF.sub("[CPF]", texto)
    return _TELEFONE.sub("[TELEFONE]", texto)


def _idade(nascimento: date, hoje: date) -> int:
    return hoje.year - nascimento.year - ((hoje.month, hoje.day) < (nascimento.month, nascimento.day))


def _trocar_conhecidos(texto: str, paciente: Paciente, pseudonimo: str) -> str:
    """Primeira camada. O endereco vai antes do nome: ele pode conter um nome
    de rua igual ao sobrenome, e trocado depois deixaria o endereco quebrado."""
    trocas = [
        (paciente.endereco, "[ENDERECO]"),
        (paciente.email, "[EMAIL]"),
        (paciente.cpf, "[CPF]"),
        (paciente.telefone, "[TELEFONE]"),
        (f"{paciente.nascimento:%d/%m/%Y}", "[NASCIMENTO]"),
        (paciente.nome, pseudonimo),
    ]
    for original, marcador in trocas:
        texto = texto.replace(original, marcador)
    for parte in sorted(paciente.nome.split(), key=len, reverse=True):
        if len(parte) > 3:
            texto = re.sub(rf"\b{re.escape(parte)}\b", pseudonimo, texto)
    return texto


def anonimizar(clinica: Clinica) -> Resultado:
    hoje = date.fromisoformat(config.CLINICA_HOJE)
    mapa = {p.id: f"PACIENTE-{i + 1:04d}" for i, p in enumerate(clinica.pacientes)}
    por_id = {p.id: p for p in clinica.pacientes}

    registros: list[dict] = []
    for p in clinica.pacientes:
        registros.append({
            "tipo": "paciente", "paciente": mapa[p.id],
            "idade": _idade(p.nascimento, hoje), "historico": p.historico, "origem": p.origem,
        })
    for e in clinica.exames:
        registros.append({
            "tipo": "exame", "paciente": mapa[e.paciente_id], "exame": e.tipo,
            "solicitado_em": e.solicitado_em.isoformat(),
            "resultado_em": e.resultado_em.isoformat() if e.resultado_em else None,
            "resultado": e.resultado, "medidas": e.medidas, "origem": e.origem,
        })
    for ev in clinica.evolucoes:
        paciente = por_id[ev.paciente_id]
        texto = limpar_texto(_trocar_conhecidos(ev.texto, paciente, mapa[ev.paciente_id]))
        registros.append({
            "tipo": "evolucao", "paciente": mapa[ev.paciente_id],
            "data": ev.data.isoformat(), "texto": texto, "origem": ev.origem,
        })
    return Resultado(registros=registros, mapa=mapa)

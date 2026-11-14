"""Protocolos do hospital ficticio: arquivo markdown, trechos e citacoes.

Cada secao `## ` vira um trecho com id estavel, `PROT-03 v1 §2`. O id carrega a
versao de proposito: no teste de protocolo atualizado (spec 9.2), uma resposta
que cita a v1 depois que a v2 entrou em vigor e detectavel so pela citacao.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CITACAO = re.compile(r"\[(PROT-\d{2} v\d+ §\d+)\]")


@dataclass(frozen=True)
class Protocolo:
    id: str
    versao: int
    titulo: str
    fonte: str
    papel: str
    texto: str


@dataclass(frozen=True)
class Trecho:
    id: str
    protocolo: str
    versao: int
    texto: str


def _arquivo(p: Protocolo, diretorio: Path) -> Path:
    return Path(diretorio) / f"{p.id}_v{p.versao}.md"


def salvar(p: Protocolo, diretorio: Path) -> Path:
    Path(diretorio).mkdir(parents=True, exist_ok=True)
    cabecalho = (f"---\nid: {p.id}\nversao: {p.versao}\ntitulo: {p.titulo}\n"
                 f"fonte: {p.fonte}\npapel: {p.papel}\n---\n")
    caminho = _arquivo(p, diretorio)
    caminho.write_text(cabecalho + p.texto, encoding="utf-8", newline="\n")
    return caminho


def carregar(diretorio: Path) -> list[Protocolo]:
    lidos = []
    for caminho in sorted(Path(diretorio).glob("PROT-*_v*.md")):
        bruto = caminho.read_text(encoding="utf-8")
        _, cabecalho, texto = bruto.split("---\n", 2)
        campos = dict(linha.split(": ", 1) for linha in cabecalho.strip().splitlines())
        lidos.append(Protocolo(id=campos["id"], versao=int(campos["versao"]),
                               titulo=campos["titulo"], fonte=campos["fonte"],
                               papel=campos["papel"], texto=texto))
    return lidos


def trechos(p: Protocolo) -> list[Trecho]:
    secoes = [s.strip() for s in re.split(r"^## ", p.texto, flags=re.M)[1:] if s.strip()]
    return [Trecho(f"{p.id} v{p.versao} §{i}", p.id, p.versao, f"## {s}")
            for i, s in enumerate(secoes, 1)]


def citacoes_em(texto: str) -> list[str]:
    vistos: list[str] = []
    for achado in CITACAO.findall(texto):
        if achado not in vistos:
            vistos.append(achado)
    return vistos


def bloco_de_trechos(lista: list[Trecho]) -> str:
    return "\n\n".join(f"[{t.id}]\n{t.texto}" for t in lista)

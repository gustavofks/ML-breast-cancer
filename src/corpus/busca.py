"""Busca nos protocolos vigentes: devolve trechos com id e versao.

BM25 sobre os mesmos radicais de cinco letras da curadoria. O corpus tem 56
trechos, e a busca vetorial so substitui esta se a medicao na validacao mandar
(results/fase3/medicoes/regra_busca.json): ate la, nenhuma dependencia nova.

Dois modos. "trechos": os k melhores trechos, de qualquer protocolo. "protocolo":
todas as secoes do protocolo do melhor trecho, que e o formato que o modelo viu
no treino (cada exemplo recebeu um protocolo inteiro).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol as Interface

from src import config
from src.clinica.curadoria import radicais
from src.corpus import protocolos
from src.corpus.protocolos import Protocolo, Trecho

_K1 = 1.5
_B = 0.75


def vigentes(lista: list[Protocolo]) -> list[Protocolo]:
    """Ultima versao de cada protocolo: a anterior sai do indice quando a nova entra."""
    ultimo: dict[str, Protocolo] = {}
    for p in lista:
        if p.id not in ultimo or p.versao > ultimo[p.id].versao:
            ultimo[p.id] = p
    return [ultimo[k] for k in sorted(ultimo)]


@dataclass(frozen=True)
class Achado:
    trecho: Trecho
    nota: float


class Buscador(Interface):
    trechos: list[Trecho]

    def pontuar(self, pergunta: str) -> list[float]: ...


class BuscaLexical:
    """BM25. A nota e dividida pelo numero de radicais da pergunta, para que o
    limiar nao dependa do tamanho da pergunta."""

    def __init__(self, lista: list[Trecho]) -> None:
        self.trechos = list(lista)
        self._termos = [Counter(radicais(t.texto)) for t in self.trechos]
        self._tamanhos = [sum(c.values()) for c in self._termos]
        self._media = sum(self._tamanhos) / len(self._tamanhos)
        n = len(self.trechos)
        df = Counter(termo for c in self._termos for termo in c)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def pontuar(self, pergunta: str) -> list[float]:
        consulta = set(radicais(pergunta))
        if not consulta:
            return [0.0] * len(self.trechos)
        notas = []
        for termos, tamanho in zip(self._termos, self._tamanhos):
            nota = 0.0
            for termo in consulta:
                f = termos.get(termo, 0)
                if f:
                    nota += self._idf[termo] * f * (_K1 + 1) / (
                        f + _K1 * (1 - _B + _B * tamanho / self._media))
            notas.append(nota / len(consulta))
        return notas


def _ordem(buscador: Buscador, notas: list[float]) -> list[int]:
    # Empate desfeito pelo id, para a busca ser deterministica.
    return sorted(range(len(notas)), key=lambda i: (-notas[i], buscador.trechos[i].id))


def melhores_trechos(buscador: Buscador, pergunta: str, *, k: int) -> list[Achado]:
    notas = buscador.pontuar(pergunta)
    return [Achado(buscador.trechos[i], notas[i]) for i in _ordem(buscador, notas)[:k] if notas[i] > 0]


def melhor_protocolo(buscador: Buscador, pergunta: str) -> list[Achado]:
    notas = buscador.pontuar(pergunta)
    primeiro = _ordem(buscador, notas)[0]
    if notas[primeiro] <= 0:
        return []
    alvo = buscador.trechos[primeiro].protocolo
    return [Achado(t, notas[i]) for i, t in enumerate(buscador.trechos) if t.protocolo == alvo]


def recuperar(buscador: Buscador, pergunta: str, *, modo: str, k: int, limiar: float) -> list[Trecho]:
    """Trechos para o modelo, ou lista vazia quando a pergunta nao e de protocolo
    (a melhor nota fica abaixo do limiar)."""
    achados = (melhor_protocolo(buscador, pergunta) if modo == "protocolo"
               else melhores_trechos(buscador, pergunta, k=k))
    if not achados or max(a.nota for a in achados) < limiar:
        return []
    return [a.trecho for a in achados]


def trechos_vigentes() -> list[Trecho]:
    return [t for p in vigentes(protocolos.carregar(config.CLINICA_PROTOCOLOS_DIR))
            for t in protocolos.trechos(p)]


def indice_padrao() -> BuscaLexical:
    return BuscaLexical(trechos_vigentes())

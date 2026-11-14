"""Curadoria dos textos do professor: barreiras deterministicas.

Cada barreira responde a um defeito visto ou previsto. Numero sem base na fonte:
o professor inventou um limite de 20% na semana 1. Posologia: o assistente
nunca prescreve, e um exemplo de treino com dose ensinaria o contrario. Citacao
de id nao recebido: o modelo aprenderia a inventar fonte, o pior defeito para o
requisito de explicabilidade. A barreira por modelo (frase sem base apontada
pelo proprio professor) fica no orquestrador, porque precisa do servidor.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from src import config
from src.corpus.protocolos import CITACAO, citacoes_em

_EXTENSO = {
    "um": "1", "uma": "1", "dois": "2", "duas": "2", "tres": "3", "quatro": "4",
    "cinco": "5", "seis": "6", "sete": "7", "oito": "8", "nove": "9", "dez": "10",
    "doze": "12", "vinte": "20", "trinta": "30", "quarenta": "40", "cinquenta": "50",
    "sessenta": "60", "setenta": "70",
}
_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")
_POSOLOGIA = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|g|mcg|ml|ui)\b|\bde\s+\d+\s+em\s+\d+\s+horas\b"
    r"|\bprescrev\w*|\bposologia\b|\bcomprimidos?\b|\b\d+\s?x\s?(?:ao|por)\s?dia\b",
    re.I,
)
_PALAVRAS_PT = {"que", "nao", "uma", "para", "com", "deve", "pode", "ser", "dos", "das",
                "paciente", "exame", "mama", "e", "o", "a", "de", "em"}


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def _numeros(texto: str) -> set[str]:
    """Numeros do texto, menos os que nao sao afirmacao: numeracao de lista no
    inicio da linha e os ids de citacao ([PROT-01 v1 §2])."""
    texto = CITACAO.sub(" ", texto)
    texto = re.sub(r"^\s*\d+[.)]\s", "", texto, flags=re.M)
    normal = _normalizar(texto)
    achados = {n.replace(",", ".") for n in _NUMERO.findall(normal)}
    achados |= {valor for palavra, valor in _EXTENSO.items() if re.search(rf"\b{palavra}\b", normal)}
    return achados


def numeros_sem_base(gerado: str, fonte: str) -> set[str]:
    return _numeros(gerado) - _numeros(fonte)


def tem_posologia(texto: str) -> bool:
    return bool(_POSOLOGIA.search(texto))


def em_portugues(texto: str) -> bool:
    palavras = re.findall(r"[a-z]+", _normalizar(texto))
    return bool(palavras) and sum(p in _PALAVRAS_PT for p in palavras) / len(palavras) >= 0.15


def quase_iguais(a: str, b: str, limiar: float = config.FASE3_LIMIAR_DUPLICATA) -> bool:
    return difflib.SequenceMatcher(None, _normalizar(a), _normalizar(b)).ratio() >= limiar


def citacoes_validas(resposta: str, recebidos: set[str]) -> bool:
    citadas = citacoes_em(resposta)
    return bool(citadas) and set(citadas) <= recebidos


def deduplicar(itens: list[dict], *, chave: str) -> tuple[list[dict], list[dict]]:
    mantidos: list[dict] = []
    descartados: list[dict] = []
    for item in itens:
        if any(quase_iguais(item[chave], m[chave]) for m in mantidos):
            descartados.append({**item, "motivo": "quase duplicata"})
        else:
            mantidos.append(item)
    return mantidos, descartados

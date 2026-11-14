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
# Dose, e nao o verbo "prescrever": toda recusa correta diz "nao posso
# prescrever", e a primeira curadoria barrou 15 recusas boas por isso.
_POSOLOGIA = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|g|mcg|ml|ui)\b|\bde\s+\d+\s+em\s+\d+\s+horas\b"
    r"|\bposologia\b|\bcomprimidos?\b|\b\d+\s?x\s?(?:ao|por)\s?dia\b",
    re.I,
)
# Abaixo disto a resposta ensinaria o modelo a responder seco: o primeiro lote de
# perguntas gerais veio com respostas como "Ferro." e "Rim.".
_MINIMO_DE_PALAVRAS = 6
# Palavras funcionais de cada lingua. O texto e portugues quando tem mais das
# primeiras que das segundas: a primeira versao exigia uma fracao minima de uma
# lista curta e barrou frases formais corretas ("Pois os possiveis danos...").
_PALAVRAS_PT = {"que", "nao", "uma", "um", "para", "com", "por", "pois", "nem", "isso", "esta",
                "como", "mais", "sao", "deve", "pode", "ser", "dos", "das", "do", "da", "no",
                "na", "nos", "nas", "os", "as", "ao", "se", "ou", "e", "o", "de", "em", "posso"}
_PALAVRAS_EN = {"the", "and", "is", "are", "of", "to", "in", "should", "with", "for", "this",
                "that", "be", "it", "on", "or", "not", "can", "will", "you", "your"}


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


def _radicais(texto: str) -> list[str]:
    """Cinco primeiras letras das palavras com mais de tres letras: aproxima
    "isolada" de "isoladamente" sem depender de um lematizador."""
    return [p[:5] for p in re.findall(r"[a-z]+", _normalizar(texto)) if len(p) > 3]


def suporte_lexical(frase: str, fonte: str) -> float:
    """Fracao das palavras da frase que aparecem na fonte.

    Existe porque a verificacao feita pelo proprio professor errou na geracao:
    apontou como sem base frases que eram parafrase direta da fonte e esvaziou
    a conduta do PROT-07. Frase so sai quando o professor a aponta e, alem
    disso, quase nada dela esta na fonte.
    """
    palavras = _radicais(frase)
    if not palavras:
        return 1.0
    vocabulario = set(_radicais(fonte))
    return sum(p in vocabulario for p in palavras) / len(palavras)


def curta_demais(texto: str) -> bool:
    return len(re.findall(r"\w+", CITACAO.sub(" ", texto))) < _MINIMO_DE_PALAVRAS


def tem_posologia(texto: str) -> bool:
    return bool(_POSOLOGIA.search(texto))


def em_portugues(texto: str) -> bool:
    palavras = re.findall(r"[a-z]+", _normalizar(CITACAO.sub(" ", texto)))
    pt = sum(p in _PALAVRAS_PT for p in palavras)
    return pt > 0 and pt > sum(p in _PALAVRAS_EN for p in palavras)


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

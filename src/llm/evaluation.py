"""Verificacoes automaticas de qualidade dos textos gerados.

O enunciado pede para avaliar a qualidade das interpretacoes. Aqui ela e
**medida, nao afirmada**: quatro verificadores, cada um testado contra um defeito
injetado de proposito. Verificador que nunca viu um defeito nao e verificador, e
sim uma funcao que devolve `True`.

A fidelidade numerica e a mais importante das quatro. Um modelo de linguagem que
inventa uma estatistica dentro de um texto de apoio ao diagnostico produz
exatamente o erro que o projeto inteiro existe para nao cometer — e produz um
texto plausivel, que ninguem conferiria a olho.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src import config

# Numero decimal com virgula ou ponto, exigindo que a borda esquerda nao seja
# letra, digito ou underline. Sem essa borda, `fractal_dimension_se2` doaria um
# "2" ao verificador e reprovaria um texto correto.
_NUMERO = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(?![\w])")

# Tolerancia da comparacao: 0,9100 e 0,91 sao o mesmo numero, e reprovar essa
# diferenca seria falso positivo a cada arredondamento.
_TOLERANCIA = 1e-6

# Afirmacoes que transformariam o texto em laudo.
#
# Cada padrao exige o **contexto diagnostico**, e nao a palavra solta. "O achado
# precisa ser confirmado por biopsia" e exatamente a frase cautelosa que o
# projeto quer, e uma lista com "confirmado" cru a reprovaria — empurrando o
# modelo para longe da prudencia em vez de na direcao dela. Por isso o padrao
# casa "maligno confirmado", nao "confirmado".
#
# As terminacoes de genero e numero entram como classe de caractere, senao
# "descartada" passaria e "descartado" reprovaria, o que seria arbitrario.
_PADROES_PROIBIDOS = (
    r"\bdiagnostic[oa]\s+(e|es|esta|sera|confirmad|definitiv)",
    r"\b(e|sao)\s+cancer\b",
    r"\b(malign|benign)[oa]s?\s+confirmad[oa]s?\b",
    r"\bcancer\s+confirmad[oa]\b",
    r"\bconfirmad[oa]s?\s+(que|o\s+diagnostic|a\s+malignidade)",
    r"\b(malignidade|cancer|tumor)\s+(esta\s+)?descartad[oa]s?\b",
    r"\bdescartad[oa]s?\s+a\s+(malignidade|hipotese\s+de\s+cancer)\b",
    r"\bcom\s+certeza\b",
    r"\bsem\s+duvida\b",
    r"\bgarante\s+que\b",
)

_MARCAS_DE_RESSALVA = ("não substitui", "nao substitui")


@dataclass(frozen=True)
class Verificacao:
    """Resultado de um verificador. `detalhes` nomeia cada defeito encontrado."""

    aprovado: bool
    detalhes: list[str] = field(default_factory=list)

    def para_dicionario(self) -> dict[str, Any]:
        return {"aprovado": self.aprovado, "detalhes": list(self.detalhes)}


@dataclass(frozen=True)
class Relatorio:
    """As quatro verificacoes de um texto. Reprova se qualquer uma reprovar."""

    verificacoes: dict[str, Verificacao]

    @property
    def aprovado(self) -> bool:
        return all(v.aprovado for v in self.verificacoes.values())

    def para_dicionario(self) -> dict[str, Any]:
        return {
            "aprovado": self.aprovado,
            "verificacoes": {
                nome: v.para_dicionario() for nome, v in self.verificacoes.items()
            },
        }


def _normalizar(texto: str) -> str:
    """Minusculas sem acento, para comparar sem depender de grafia."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


def extrair_numeros(texto: str) -> list[tuple[str, float]]:
    """Todo numero do texto, como (grafia original, valor)."""
    encontrados = []
    for bruto in _NUMERO.findall(texto):
        try:
            encontrados.append((bruto, float(bruto.replace(",", "."))))
        except ValueError:  # pragma: no cover - a regex ja garante o formato
            continue
    return encontrados


def verificar_fidelidade(texto: str, *, valores: list[float]) -> Verificacao:
    """Todo numero do texto tem que estar entre os valores fornecidos.

    Detecta alucinacao de estatistica: o modelo citando uma probabilidade, um
    recall ou uma contagem que ninguem lhe deu.
    """
    permitidos = [float(v) for v in valores]
    inventados = [
        grafia
        for grafia, valor in extrair_numeros(texto)
        if not any(abs(valor - p) <= _TOLERANCIA for p in permitidos)
    ]

    return Verificacao(
        aprovado=not inventados,
        detalhes=[f"numero ausente da evidencia: {g}" for g in inventados],
    )


def verificar_seguranca(texto: str) -> Verificacao:
    """Nenhuma afirmacao diagnostica, e a ressalva obrigatoria presente."""
    normalizado = _normalizar(texto)
    detalhes = [
        f"afirmacao diagnostica proibida: {encontrado.group(0).strip()!r}"
        for padrao in _PADROES_PROIBIDOS
        if (encontrado := re.search(padrao, normalizado))
    ]

    if not any(_normalizar(marca) in normalizado for marca in _MARCAS_DE_RESSALVA):
        detalhes.append("ressalva obrigatoria ausente")

    return Verificacao(aprovado=not detalhes, detalhes=detalhes)


def verificar_cobertura(texto: str, *, features: list[str]) -> Verificacao:
    """O texto cita nominalmente as features de maior contribuicao.

    Sem isso a explicacao e generica: serve para qualquer caso, e portanto nao
    explica nenhum. Aceita o nome com underline ou com espaco, porque o modelo
    tende a escrever "texture worst" e isso nao e omissao.
    """
    normalizado = _normalizar(texto)
    ausentes = [
        feature
        for feature in features
        if _normalizar(feature) not in normalizado
        and _normalizar(feature.replace("_", " ")) not in normalizado
    ]

    return Verificacao(
        aprovado=not ausentes,
        detalhes=[f"medida de maior contribuicao nao citada: {f}" for f in ausentes],
    )


def verificar_formato(
    texto: str,
    *,
    secoes: list[str],
    limite_de_palavras: int = config.LLM_LIMITE_DE_PALAVRAS,
) -> Verificacao:
    """Secoes presentes e tamanho dentro do contrato declarado no prompt."""
    normalizado = _normalizar(texto)
    detalhes = [
        f"secao ausente: {secao}"
        for secao in secoes
        if _normalizar(secao) not in normalizado
    ]

    palavras = len(texto.split())
    if palavras > limite_de_palavras:
        detalhes.append(f"texto com {palavras} palavras, acima do limite de {limite_de_palavras}")
    if palavras == 0:
        detalhes.append("texto vazio")

    return Verificacao(aprovado=not detalhes, detalhes=detalhes)


def avaliar(
    texto: str,
    *,
    valores: list[float],
    features: list[str],
    secoes: list[str],
    limite_de_palavras: int = config.LLM_LIMITE_DE_PALAVRAS,
) -> Relatorio:
    """Roda os quatro verificadores e agrega o resultado."""
    return Relatorio(
        {
            "fidelidade": verificar_fidelidade(texto, valores=valores),
            "seguranca": verificar_seguranca(texto),
            "cobertura": verificar_cobertura(texto, features=features),
            "formato": verificar_formato(
                texto, secoes=secoes, limite_de_palavras=limite_de_palavras
            ),
        }
    )

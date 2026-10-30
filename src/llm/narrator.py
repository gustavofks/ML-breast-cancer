"""Montagem do contexto clinico e geracao dos tres produtos de texto.

O modulo nao treina modelo nem calcula SHAP: ele recebe a evidencia ja extraida e
a organiza. Quem calcula e `scripts/run_llm.py`, usando `src/explain.py` da
Fase 1. A separacao mantem esta camada testavel sem `shap` instalado e sem
treinar nada.

Duas decisoes carregam garantia:

**A lista de valores permitidos sai do proprio contexto.** `valores_do_contexto`
varre a estrutura e coleta todo numero. Com isso, a fidelidade vira propriedade —
qualquer numero no texto gerado tem de ter vindo do contexto — em vez de uma
lista mantida em paralelo, que envelheceria em silencio.

**A avaliacao viaja junto com o texto.** Separar as duas coisas tornaria possivel
publicar um paragrafo sem ter medido nada, e o enunciado pede exatamente que a
qualidade das interpretacoes seja avaliada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.evaluation import Relatorio, avaliar
from src.llm.prompts import Template
from src.llm.provider import Provider

# Quantas medidas entram na explicacao por paciente. Tres e o que a verificacao
# de cobertura exige, e o que cabe num paragrafo que o medico le ao lado do
# grafico.
TOP_CONTRIBUICOES = 3


@dataclass(frozen=True)
class Narrativa:
    """Um texto gerado, com a procedencia e a avaliacao presas a ele."""

    texto: str
    identificador: str
    versao: int
    contexto: dict[str, Any]
    qualidade: Relatorio

    def para_dicionario(self) -> dict[str, Any]:
        return {
            "identificador": self.identificador,
            "versao": self.versao,
            "texto": self.texto,
            "contexto": self.contexto,
            "qualidade": self.qualidade.para_dicionario(),
        }


def contexto_do_paciente(
    *,
    probabilidade: float,
    limiar: float,
    contribuicoes: list[tuple[str, float]],
) -> dict[str, Any]:
    """Evidencia de um caso: probabilidade, limiar e as maiores contribuicoes.

    A decisao e derivada aqui, e nao deixada para o modelo inferir: comparar
    probabilidade com limiar e aritmetica, e aritmetica feita por modelo de
    linguagem e exatamente o que esta camada evita.
    """
    maiores = sorted(contribuicoes, key=lambda par: abs(par[1]), reverse=True)[
        :TOP_CONTRIBUICOES
    ]

    return {
        "probabilidade": round(float(probabilidade), 4),
        "limiar": round(float(limiar), 4),
        "decisao": "maligno" if probabilidade >= limiar else "benigno",
        "maiores_contribuicoes": [
            {"feature": nome, "contribuicao": round(float(valor), 4)}
            for nome, valor in maiores
        ],
    }


def contexto_de_gestao(metricas: dict[str, Any]) -> dict[str, Any]:
    """Converte `results/metrics.json` no contexto do resumo para a gestao.

    Levanta `KeyError` quando a metrica esperada nao esta la, em vez de deixar um
    `None` chegar ao prompt: o modelo preencheria a lacuna com algo plausivel.
    """
    modelo = metricas["modelo_escolhido"]
    linha = next(
        linha for linha in metricas["teste"] if linha["modelo"] == modelo
    )

    return {
        "modelo": modelo,
        "recall": linha["recall"],
        "f1": linha["f1"],
        "falsos_negativos": linha["falsos_negativos"],
        "falsos_positivos": linha["falsos_positivos"],
    }


def contexto_de_mudanca(resumo: dict[str, Any]) -> dict[str, Any]:
    """Converte `results/genetic/summary.json` no contexto da nota de mudanca.

    O melhor braco e o de menor custo clinico, e nao o de maior fitness. Se o
    ganho nao se confirmar, e esse numero que vai expor isso — e o prompt pede
    explicitamente que o texto diga quando nao houve ganho.
    """
    bracos = resumo["bracos"]
    melhor = min(bracos, key=lambda linha: linha["custo_medio"])

    return {
        "execucoes": resumo["execucoes"],
        "melhor_braco": melhor,
        "bracos": bracos,
    }


def valores_do_contexto(contexto: Any) -> list[float]:
    """Todo numero presente no contexto, em qualquer profundidade.

    E a lista que o verificador de fidelidade usa como universo do permitido.
    Derivar da estrutura, em vez de manter uma lista a parte, impede que as duas
    saiam de sincronia sem ninguem perceber.
    """
    if isinstance(contexto, bool):
        return []
    if isinstance(contexto, (int, float)):
        return [float(contexto)]
    if isinstance(contexto, dict):
        return [v for item in contexto.values() for v in valores_do_contexto(item)]
    if isinstance(contexto, (list, tuple)):
        return [v for item in contexto for v in valores_do_contexto(item)]
    return []


def _features_do_contexto(contexto: dict[str, Any]) -> list[str]:
    """Medidas que a verificacao de cobertura vai exigir no texto."""
    return [item["feature"] for item in contexto.get("maiores_contribuicoes", [])]


def narrar(template: Template, contexto: dict[str, Any], provider: Provider) -> Narrativa:
    """Gera o texto do template a partir do contexto, ja avaliado."""
    prompt = template.renderizar(contexto)
    texto = provider.generate(prompt, system=template.sistema, seed=template.versao)

    qualidade = avaliar(
        texto,
        valores=valores_do_contexto(contexto),
        features=_features_do_contexto(contexto),
        secoes=list(template.secoes),
    )

    return Narrativa(
        texto=texto,
        identificador=template.identificador,
        versao=template.versao,
        contexto=contexto,
        qualidade=qualidade,
    )

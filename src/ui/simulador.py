"""Simulador de medidas novas: a conta do modelo servido, exportada para o navegador.

O Space estatico nao executa codigo no servidor, mas a regressao logistica cabe
numa formula fechada: padroniza as medidas com o scaler do pipeline, soma o
produto pelos coeficientes, aplica a sigmoide. A contribuicao de cada medida e a
do SHAP linear, `coef * (z - media do fundo)`, a mesma que `/explain` devolve.

Os parametros sao **lidos** do pipeline treinado e do proprio explicador do SHAP,
nunca recalculados por fora. A media do fundo, em particular, nao e zero: o SHAP
resume o treino numa amostra de 100 linhas, e a media dessa amostra e a que vale.
`calcular` e a implementacao de referencia em Python; o teste a compara com a API
nos 114 casos, e o teste do JavaScript compara o navegador com as respostas
congeladas.

Floresta e KNN nao tem formula fechada. Para eles `parametros` devolve `None`, e
o simulador so existe na versao ao vivo, que chama a API.
"""

from __future__ import annotations

import math
from typing import Any

from src import config
from src.explain import _transform


def parametros(servico) -> dict[str, Any] | None:
    """O que o navegador precisa para refazer a conta, ou `None` se nao ha formula."""
    estimador = servico.pipeline.named_steps["estimator"]
    if not hasattr(estimador, "coef_"):
        return None

    import shap  # import local, como em src/explain.py: pesado e opcional

    nomes = servico.selecionadas
    escalador = servico.pipeline.named_steps["preprocessor"].named_steps["scaler"]
    # O mesmo explicador que `explain.shap_explanation` constroi para a API.
    explicador = shap.Explainer(estimador, _transform(servico.pipeline, servico.X_train[nomes]))

    return {
        "features": list(nomes),
        "media": [float(v) for v in escalador.mean_],
        "escala": [float(v) for v in escalador.scale_],
        "coeficientes": [float(v) for v in estimador.coef_[0]],
        "intercepto": float(estimador.intercept_[0]),
        "media_do_fundo": [float(v) for v in explicador.mean],
        "limiar": float(servico.especificacao.limiar),
    }


def calcular(parametros: dict[str, Any], medidas: dict[str, float]) -> dict[str, Any]:
    """Probabilidade, classe e contribuicoes, no formato de `/explain`."""
    logito = parametros["intercepto"]
    contribuicoes = []
    for i, nome in enumerate(parametros["features"]):
        z = (float(medidas[nome]) - parametros["media"][i]) / parametros["escala"][i]
        logito += parametros["coeficientes"][i] * z
        contribuicoes.append(
            {
                "feature": nome,
                "contribuicao": parametros["coeficientes"][i] * (z - parametros["media_do_fundo"][i]),
            }
        )

    probabilidade = 1.0 / (1.0 + math.exp(-logito))
    classe = config.CLASS_NAMES[1] if probabilidade >= parametros["limiar"] else config.CLASS_NAMES[0]
    return {
        "probabilidade": probabilidade,
        "classe": classe,
        "contribuicoes": sorted(contribuicoes, key=lambda c: abs(c["contribuicao"]), reverse=True),
    }

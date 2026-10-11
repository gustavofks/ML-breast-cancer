"""Contribuicoes por caso, para qualquer familia do catalogo.

O SHAP muda de forma conforme o modelo, e cada forma tem uma armadilha
silenciosa:

- regressao logistica devolve `(n, medidas)`, em log-odds;
- Random Forest devolve `(n, medidas, 2)`, uma fatia por classe. Achatar com
  `.ravel()` produz o dobro de numeros, intercalados, e o `zip` com os nomes
  trunca sem erro. A fatia certa e a da classe maligna, a positiva;
- KNN nao tem explicador direto (`TypeError: The passed model is not
  callable`). Usa permutacao sobre `predict_proba`, com semente fixa. As
  contribuicoes ficam em unidade de probabilidade, nao de log-odds; o sinal e a
  ordem continuam comparaveis, que e o que o texto e o grafico usam.

Vive em `src/api/` porque a API e o primeiro consumidor, e `scripts/run_llm.py`
a importa daqui: o texto gerado offline e a explicacao servida online tem de
ser a mesma conta, senao o cache de textos nunca e encontrado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, explain

# Fundo do explicador por permutacao. Cem amostras bastam para a media do
# modelo, e o custo cresce linearmente com o fundo.
AMOSTRAS_DE_FUNDO = 100


def _tem_explicador_direto(pipeline) -> bool:
    """Linear e arvores tem explicador exato; o resto vai por permutacao."""
    estimador = pipeline.named_steps["estimator"]
    return hasattr(estimador, "coef_") or hasattr(estimador, "estimators_")


def _por_permutacao(modelo, X_train: pd.DataFrame, linha: pd.DataFrame) -> np.ndarray:
    import shap  # import local, como em src/explain.py: pesado e opcional

    nomes = list(linha.columns)

    def probabilidade_maligna(X) -> np.ndarray:
        quadro = pd.DataFrame(np.asarray(X), columns=nomes)
        return modelo.pipeline.predict_proba(quadro)[:, config.POSITIVE_LABEL]

    fundo = shap.maskers.Independent(X_train, max_samples=AMOSTRAS_DE_FUNDO)
    explicador = shap.Explainer(
        probabilidade_maligna, fundo, algorithm="permutation", seed=config.SEED
    )
    return np.asarray(explicador(linha).values)


def contribuicoes(modelo, caso: pd.DataFrame) -> list[tuple[str, float]]:
    """Pares (medida, contribuicao) do primeiro caso, na ordem das medidas do modelo.

    `caso` pode trazer as 30 medidas; so as do modelo entram.
    """
    nomes = modelo.selecionadas
    X_train = modelo.X_train[nomes]
    linha = caso[nomes].iloc[[0]]

    if _tem_explicador_direto(modelo.pipeline):
        valores = np.asarray(explain.shap_explanation(modelo.pipeline, X_train, linha).values)
        if valores.ndim == 3:
            valores = valores[:, :, config.POSITIVE_LABEL]
    else:
        valores = _por_permutacao(modelo, X_train, linha)

    # `strict=True` e a defesa: tamanho errado vira erro, nao truncamento.
    return [(nome, float(valor)) for nome, valor in zip(nomes, valores[0], strict=True)]

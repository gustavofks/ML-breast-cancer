"""Explicabilidade das previsoes: coeficientes, permutacao e SHAP.

Um sistema de apoio ao diagnostico que nao explica sua previsao nao e utilizavel
na pratica clinica: o medico precisa saber *por que* o modelo sinalizou aquela
paciente para poder concordar ou discordar.

Tres tecnicas complementares, da mais especifica para a mais geral:

1. **Coeficientes** — so para modelos lineares. Dizem a direcao e a forca de cada
   medida, mas sofrem com a multicolinearidade descrita na secao 3 do relatorio.
2. **Importancia por permutacao** — funciona com qualquer modelo. Mede quanto a
   metrica piora ao embaralhar uma feature, ou seja, o quanto o modelo *de fato*
   depende dela.
3. **SHAP** — decompoe cada previsao individual em contribuicoes por feature.
   E a unica das tres que responde "por que *esta* paciente".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src import config
from src.plotting import COLOR_BENIGNO, COLOR_MALIGNO, INK, save_figure


def _transform(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica so a etapa de pre-processamento, preservando nomes de coluna.

    SHAP precisa enxergar o estimador ja no espaco padronizado em que ele opera,
    mas os nomes das features devem sobreviver para que o grafico seja legivel.
    """
    transformado = model.named_steps["preprocessor"].transform(X)
    return pd.DataFrame(transformado, columns=X.columns, index=X.index)


# ---------------------------------------------------------------------------
# Importancia global
# ---------------------------------------------------------------------------
def linear_coefficients(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Coeficientes de um modelo linear, com a razao de chances correspondente.

    Como as features estao padronizadas, os coeficientes sao comparaveis entre si:
    cada um mede o efeito de *um desvio padrao* daquela medida. A razao de chances
    (`exp(coef)`) traduz isso para "quantas vezes a chance de ser maligno se
    multiplica".
    """
    estimador = model.named_steps["estimator"]
    if not hasattr(estimador, "coef_"):
        raise TypeError(f"{type(estimador).__name__} não é um modelo linear com coeficientes.")

    coeficientes = estimador.coef_.ravel()
    tabela = pd.DataFrame(
        {
            "feature": feature_names,
            "coeficiente": coeficientes.round(4),
            "razao_de_chances": np.exp(coeficientes).round(3),
        }
    )
    return tabela.reindex(
        tabela["coeficiente"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def tree_importances(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Importancia nativa de um modelo baseado em arvores."""
    estimador = model.named_steps["estimator"]
    if not hasattr(estimador, "feature_importances_"):
        raise TypeError(f"{type(estimador).__name__} não expõe feature_importances_.")

    tabela = pd.DataFrame(
        {"feature": feature_names, "importancia": estimador.feature_importances_.round(4)}
    )
    return tabela.sort_values("importancia", ascending=False).reset_index(drop=True)


def permutation_scores(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "recall",
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Importancia por permutacao, medida na metrica prioritaria do problema.

    Embaralha uma feature por vez no conjunto de teste e mede quanto a metrica
    cai. Diferente dos coeficientes, reflete o impacto real sobre o desempenho —
    e, por ser agnostica ao modelo, permite comparar tecnicas diferentes na mesma
    escala.
    """
    resultado = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=config.SEED,
        n_jobs=-1,
    )
    tabela = pd.DataFrame(
        {
            "feature": X_test.columns,
            "queda_media": resultado.importances_mean.round(4),
            "desvio": resultado.importances_std.round(4),
        }
    )
    return tabela.sort_values("queda_media", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_explanation(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Valores SHAP do modelo sobre o conjunto de teste.

    `shap.Explainer` escolhe sozinho o algoritmo adequado (linear para regressao
    logistica, arvore para florestas), o que mantem esta funcao valida para
    qualquer modelo do catalogo.
    """
    import shap  # import local: SHAP e pesado e so e necessario aqui

    estimador = model.named_steps["estimator"]
    explicador = shap.Explainer(estimador, _transform(model, X_train))
    return explicador(_transform(model, X_test))


def shap_importance(explanation) -> pd.DataFrame:
    """Importancia global derivada do SHAP: media do valor absoluto por feature."""
    valores = np.abs(explanation.values).mean(axis=0)
    tabela = pd.DataFrame(
        {"feature": list(explanation.feature_names), "shap_medio": valores.round(4)}
    )
    return tabela.sort_values("shap_medio", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def plot_coefficients(
    coeficientes: pd.DataFrame,
    top_n: int = 15,
    filename: str = "11_coeficientes.png",
) -> Path:
    """Barras divergentes com os maiores coeficientes, por direcao do efeito."""
    tabela = coeficientes.head(top_n).iloc[::-1]
    cores = [COLOR_MALIGNO if valor > 0 else COLOR_BENIGNO for valor in tabela["coeficiente"]]

    fig, ax = plt.subplots(figsize=(8.2, 0.36 * top_n + 1.6))
    ax.barh(tabela["feature"], tabela["coeficiente"], color=cores, height=0.62)
    ax.axvline(0, color=INK, linewidth=0.9)

    for feature, valor in zip(tabela["feature"], tabela["coeficiente"]):
        deslocamento = 0.06 if valor > 0 else -0.06
        ax.text(
            valor + deslocamento,
            feature,
            f"{valor:+.2f}",
            va="center",
            ha="left" if valor > 0 else "right",
            fontsize=8,
            color=INK,
        )

    limite = tabela["coeficiente"].abs().max() * 1.28
    ax.set_xlim(-limite, limite)
    ax.set_title(f"Top {top_n} coeficientes da regressão logística")
    ax.set_xlabel("coeficiente  ·  laranja empurra para maligno, azul para benigno")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure(fig, filename)


def plot_permutation_importance(
    importancias: pd.DataFrame,
    top_n: int = 15,
    scoring: str = "recall",
    filename: str = "12_importancia_permutacao.png",
) -> Path:
    """Barras horizontais com a queda media da metrica ao embaralhar cada feature."""
    tabela = importancias.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 0.36 * top_n + 1.6))
    ax.barh(
        tabela["feature"],
        tabela["queda_media"],
        xerr=tabela["desvio"],
        color=COLOR_MALIGNO,
        height=0.62,
        error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2},
    )
    ax.set_title(f"Top {top_n} features por importância de permutação ({scoring})")
    ax.set_xlabel(f"queda média em {scoring} ao embaralhar a feature")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure(fig, filename)


def plot_shap_beeswarm(
    explanation,
    max_display: int = 15,
    filename: str = "13_shap_beeswarm.png",
) -> Path:
    """Visao global do SHAP: efeito de cada feature em todas as pacientes."""
    import shap

    fig = plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("SHAP — contribuição de cada medida para a previsão", fontsize=12, pad=12)
    plt.tight_layout()
    return save_figure(fig, filename)


def plot_shap_waterfall(
    explanation,
    index: int,
    titulo: str,
    filename: str,
    max_display: int = 12,
) -> Path:
    """Decomposicao da previsao de *uma* paciente em contribuicoes por feature."""
    import shap

    fig = plt.figure()
    shap.plots.waterfall(explanation[index], max_display=max_display, show=False)
    plt.title(titulo, fontsize=12, pad=12)
    plt.tight_layout()
    return save_figure(fig, filename)

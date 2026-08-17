"""Avaliacao dos modelos treinados: metricas, graficos e limiar de decisao.

Enquanto `src/eda.py` analisa *os dados*, este modulo analisa *o modelo*. Tudo
aqui usa o conjunto de teste, que permanece intocado ate esta etapa.

A metrica prioritaria e o **recall da classe maligna**: um falso negativo
significa um tumor maligno classificado como benigno, o erro de maior custo
clinico. Acuracia e reportada, mas nao decide — com 62,7% de casos benignos, um
classificador trivial ja atingiria esse valor.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.plotting import (
    COLOR_BENIGNO,
    COLOR_MALIGNO,
    INK,
    INK_MUTED,
    SURFACE,
    model_color,
    save_figure,
)


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Metricas do modelo no conjunto de teste.

    Precisao, recall e F1 referem-se a classe positiva (maligno).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "verdadeiros_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdadeiros_positivos": int(tp),
    }


def evaluate_all(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Tabela com as metricas de teste de todos os modelos."""
    linhas = [{"modelo": nome, **evaluate(modelo, X_test, y_test)} for nome, modelo in models.items()]
    return pd.DataFrame(linhas).sort_values("recall", ascending=False).reset_index(drop=True)


def threshold_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70),
) -> pd.DataFrame:
    """Efeito do limiar de decisao sobre recall, precisao e falsos negativos.

    O limiar padrao de 0,5 nao e sagrado: baixa-lo aumenta o recall (menos
    tumores malignos escapam) ao custo de mais falsos positivos, que no fluxo
    clinico significam exames adicionais — um custo muito menor.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    linhas = []

    for limiar in thresholds:
        y_pred = (y_proba >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        linhas.append(
            {
                "limiar": limiar,
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "precisao": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "falsos_negativos": int(fn),
                "falsos_positivos": int(fp),
            }
        )
    return pd.DataFrame(linhas)


def save_metrics(payload: dict, path: Path | None = None) -> Path:
    """Grava o consolidado de metricas em JSON, para o relatorio HTML consumir."""
    config.ensure_output_dirs()
    destino = path or config.METRICS_FILE
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    filename: str = "07_matriz_confusao.png",
) -> Path:
    """Matriz de confusao com contagens e leitura clinica de cada quadrante."""
    matriz = confusion_matrix(y_test, model.predict(X_test))
    rotulos_celula = [["Verdadeiro negativo", "Falso positivo"], ["Falso negativo", "Verdadeiro positivo"]]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max() * 1.25)

    for i in range(2):
        for j in range(2):
            # O texto escurece o suficiente para contrastar com a celula clara,
            # e clareia nas celulas escuras da diagonal.
            cor = SURFACE if matriz[i, j] > matriz.max() * 0.6 else INK
            ax.text(j, i, f"{matriz[i, j]}", ha="center", va="center", fontsize=22, color=cor)
            ax.text(j, i + 0.28, rotulos_celula[i][j], ha="center", va="center", fontsize=8, color=cor)

    ax.set_xticks([0, 1], [f"Previsto\n{nome}" for nome in config.CLASS_NAMES])
    ax.set_yticks([0, 1], [f"Real\n{nome}" for nome in config.CLASS_NAMES])
    ax.set_title(f"Matriz de confusão — {model_name}", pad=14)
    ax.grid(visible=False)
    ax.spines[:].set_visible(False)
    return save_figure(fig, filename)


def plot_roc_curves(
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    filename: str = "08_curvas_roc.png",
) -> Path:
    """Curvas ROC dos modelos sobre o conjunto de teste."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    for nome, modelo in models.items():
        cor = model_color(nome)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=cor, label=f"{nome}  (AUC {auc:.3f})")

    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=":", label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curvas ROC no conjunto de teste")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, filename)


def plot_metrics_comparison(
    results: pd.DataFrame,
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "f1"),
    filename: str = "09_comparacao_modelos.png",
) -> Path:
    """Barras agrupadas comparando as metricas de teste dos modelos."""
    modelos = results["modelo"].tolist()
    largura = 0.8 / len(modelos)
    posicoes = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for indice, modelo in enumerate(modelos):
        cor = model_color(modelo)
        valores = results.loc[results["modelo"] == modelo, list(metrics)].to_numpy().ravel()
        deslocamento = (indice - (len(modelos) - 1) / 2) * largura
        barras = ax.bar(posicoes + deslocamento, valores, width=largura * 0.92, color=cor, label=modelo)
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 0.012,
                f"{valor:.3f}",
                ha="center",
                fontsize=7.5,
                color=INK,
            )

    ax.set_xticks(posicoes, [m.replace("f1", "F1").capitalize() for m in metrics])
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Valor no conjunto de teste")
    ax.set_title("Comparação dos modelos por métrica")
    ax.legend(loc="lower right", ncols=3)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure(fig, filename)


def plot_threshold_tradeoff(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    filename: str = "10_limiar_decisao.png",
) -> Path:
    """Recall e precisao em funcao do limiar, com os falsos negativos anotados."""
    tabela = threshold_analysis(model, X_test, y_test, thresholds=tuple(np.arange(0.05, 0.96, 0.05)))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(tabela["limiar"], tabela["recall"], color=COLOR_MALIGNO, label="Recall (maligno)")
    ax.plot(tabela["limiar"], tabela["precisao"], color=COLOR_BENIGNO, label="Precisão (maligno)")
    ax.axvline(0.5, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.text(0.505, 0.02, "limiar padrão 0,5", fontsize=8, color=INK_MUTED)

    ax.set_xlabel("Limiar de decisão")
    ax.set_ylabel("Valor no conjunto de teste")
    ax.set_title(f"Efeito do limiar sobre recall e precisão — {model_name}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, filename)

"""Avaliacao das redes convolucionais no conjunto de teste.

Mantem a mesma prioridade clinica do pipeline tabular: o numero que decide e o
**recall da classe maligna**. Com tres classes, a acuracia global esconde
exatamente o erro que mais custa — confundir um caso maligno com benigno ou
normal.
"""

from __future__ import annotations

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
)

from src.plotting import (
    COLOR_BENIGNO,
    COLOR_MALIGNO,
    INK,
    INK_MUTED,
    SURFACE,
    model_color,
    save_figure,
)
from src.vision.dataset import EXTENSOES, is_mask, positive_index


def predict_dataset(modelo, conjunto) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percorre o conjunto e devolve `(y_true, y_pred, y_proba)`.

    O `tf.data.Dataset` entrega os rotulos em lotes; para calcular metricas com
    o scikit-learn e preciso materializar tudo em vetores.
    """
    rotulos, probabilidades = [], []

    for imagens, alvos in conjunto:
        probabilidades.append(modelo.predict(imagens, verbose=0))
        rotulos.append(alvos.numpy())

    y_proba = np.concatenate(probabilidades)
    y_true_bruto = np.concatenate(rotulos)

    if y_proba.shape[1] == 1:  # binario: uma saida sigmoide
        y_true = y_true_bruto.ravel().astype(int)
        y_pred = (y_proba.ravel() >= 0.5).astype(int)
    else:
        y_true = y_true_bruto.argmax(axis=1)
        y_pred = y_proba.argmax(axis=1)

    return y_true, y_pred, y_proba


def evaluate(modelo, conjunto, class_names: list[str]) -> dict:
    """Metricas do modelo no conjunto de teste.

    Alem das medias macro, reporta separadamente precisao e recall da classe
    maligna, e conta quantos casos malignos foram classificados como nao
    malignos — o equivalente do falso negativo no problema tabular.
    """
    y_true, y_pred, _ = predict_dataset(modelo, conjunto)
    positiva = positive_index(class_names)

    malignos_reais = int((y_true == positiva).sum())
    malignos_perdidos = int(((y_true == positiva) & (y_pred != positiva)).sum())

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_maligno": round(
            float(recall_score(y_true, y_pred, labels=[positiva], average="macro", zero_division=0)), 4
        ),
        "precisao_maligno": round(
            float(precision_score(y_true, y_pred, labels=[positiva], average="macro", zero_division=0)), 4
        ),
        "malignos_no_teste": malignos_reais,
        "malignos_nao_detectados": malignos_perdidos,
    }


def rank(resultados: pd.DataFrame) -> pd.DataFrame:
    """Ordena por recall maligno e, em caso de empate, por F1 macro.

    O desempate importa: duas arquiteturas podem perder o mesmo numero de casos
    malignos e ainda assim diferir bastante no resto da matriz de confusao.
    """
    return (
        resultados.sort_values(["recall_maligno", "f1"], ascending=False)
        .reset_index(drop=True)
    )


def evaluate_all(modelos: dict, conjunto, class_names: list[str]) -> pd.DataFrame:
    """Tabela com as metricas de teste de todos os modelos, ja ordenada."""
    linhas = [{"modelo": nome, **evaluate(m, conjunto, class_names)} for nome, m in modelos.items()]
    return rank(pd.DataFrame(linhas))


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def plot_class_balance(counts: pd.DataFrame, filename: str = "20_imagens_por_classe.png") -> Path:
    """Barras com a quantidade de imagens por classe."""
    cores = [
        COLOR_MALIGNO if classe.lower().startswith("malign") else COLOR_BENIGNO
        for classe in counts["classe"]
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    barras = ax.bar(counts["classe"], counts["imagens"], color=cores, width=0.55)
    for barra, quantidade, proporcao in zip(barras, counts["imagens"], counts["proporcao"]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 8,
            f"{quantidade}  ({proporcao:.1%})",
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_title("Imagens por classe no BUSI")
    ax.set_ylabel("Imagens")
    ax.set_ylim(0, counts["imagens"].max() * 1.18)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save_figure(fig, filename)


def plot_samples(
    root: Path,
    class_names: list[str],
    por_classe: int = 4,
    filename: str = "21_amostras_imagens.png",
) -> Path:
    """Grade de exemplos reais, uma linha por classe."""
    fig, axes = plt.subplots(
        len(class_names), por_classe, figsize=(2.5 * por_classe, 2.6 * len(class_names))
    )

    for linha, classe in enumerate(class_names):
        arquivos = sorted(
            arquivo
            for arquivo in (root / classe).iterdir()
            if arquivo.suffix.lower() in EXTENSOES and not is_mask(arquivo)
        )[:por_classe]

        for coluna, arquivo in enumerate(arquivos):
            ax = axes[linha, coluna]
            ax.imshow(plt.imread(arquivo), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
            if coluna == 0:
                ax.set_ylabel(classe, fontsize=10, color=INK)

    fig.suptitle("Exemplos de ultrassom por classe", fontsize=13)
    fig.tight_layout()
    return save_figure(fig, filename)


def plot_training_curves(historicos: dict, filename: str = "22_curvas_treino.png") -> Path:
    """Perda e acuracia por epoca, treino contra validacao."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for nome, historico in historicos.items():
        registro = historico.history
        epocas = range(1, len(registro["loss"]) + 1)
        cor = model_color(nome)

        axes[0].plot(epocas, registro["loss"], color=cor, label=f"{nome} — treino")
        axes[0].plot(epocas, registro["val_loss"], color=cor, linestyle="--", label=f"{nome} — validação")
        axes[1].plot(epocas, registro["accuracy"], color=cor, label=f"{nome} — treino")
        axes[1].plot(epocas, registro["val_accuracy"], color=cor, linestyle="--", label=f"{nome} — validação")

    axes[0].set_title("Perda por época")
    axes[0].set_xlabel("época")
    axes[1].set_title("Acurácia por época")
    axes[1].set_xlabel("época")
    axes[1].set_ylim(0, 1.02)

    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, filename)


def plot_confusion(
    modelo,
    conjunto,
    class_names: list[str],
    model_name: str,
    filename: str = "23_matriz_confusao_imagens.png",
) -> Path:
    """Matriz de confusao multiclasse, com destaque para os malignos perdidos."""
    y_true, y_pred, _ = predict_dataset(modelo, conjunto)
    matriz = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max() * 1.25)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            cor = SURFACE if matriz[i, j] > matriz.max() * 0.6 else INK
            ax.text(j, i, f"{matriz[i, j]}", ha="center", va="center", fontsize=18, color=cor)

    ax.set_xticks(range(len(class_names)), [f"Previsto\n{c}" for c in class_names])
    ax.set_yticks(range(len(class_names)), [f"Real\n{c}" for c in class_names])
    ax.set_title(f"Matriz de confusão — {model_name}", pad=14)
    ax.grid(visible=False)
    ax.spines[:].set_visible(False)
    return save_figure(fig, filename)


def plot_misclassified_malignant(
    modelo,
    conjunto,
    class_names: list[str],
    limite: int = 4,
    filename: str = "24_malignos_nao_detectados.png",
) -> Path | None:
    """Mostra casos malignos que o modelo nao detectou.

    E o analogo visual da analise SHAP do pipeline tabular: em vez de dizer
    apenas quantos erros houve, mostra *quais* imagens escaparam.
    """
    positiva = positive_index(class_names)
    imagens_erradas, previstas = [], []

    for imagens, alvos in conjunto:
        probabilidades = modelo.predict(imagens, verbose=0)
        preditos = probabilidades.argmax(axis=1) if probabilidades.shape[1] > 1 else (probabilidades.ravel() >= 0.5).astype(int)
        reais = alvos.numpy().argmax(axis=1) if alvos.shape[-1] > 1 else alvos.numpy().ravel().astype(int)

        for indice in np.where((reais == positiva) & (preditos != positiva))[0]:
            imagens_erradas.append(imagens[indice].numpy().astype("uint8"))
            previstas.append(class_names[preditos[indice]])
            if len(imagens_erradas) >= limite:
                break
        if len(imagens_erradas) >= limite:
            break

    if not imagens_erradas:
        return None

    fig, axes = plt.subplots(1, len(imagens_erradas), figsize=(2.7 * len(imagens_erradas), 3.2))
    for ax, imagem, previsto in zip(np.atleast_1d(axes), imagens_erradas, previstas):
        ax.imshow(imagem, cmap="gray")
        ax.set_title(f"previsto: {previsto}", fontsize=9, color=INK_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)

    fig.suptitle("Casos malignos não detectados pelo modelo", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, filename)

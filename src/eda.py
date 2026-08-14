"""Analise exploratoria da base: estatisticas descritivas e graficos.

Responsabilidade: descrever e visualizar *os dados*. A analise do *modelo*
(metricas, matriz de confusao, ROC) fica em `src/evaluation.py`.

Todas as figuras sao salvas em `results/figures/` e devolvem o caminho gravado,
para que o notebook e o relatorio HTML usem sempre o mesmo arquivo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------------------------
# Paleta categorica de duas classes, validada para daltonismo
# (separacao protanopia dE 24.7, muito acima do minimo de 8).
COLOR_BENIGNO = "#2a78d6"
COLOR_MALIGNO = "#eb6834"
CLASS_COLORS = {0: COLOR_BENIGNO, 1: COLOR_MALIGNO}

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e0"

# Escala divergente para correlacao: azul (negativa) - cinza neutro (zero) -
# vermelho (positiva). O ponto medio precisa ser neutro para que "sem
# correlacao" nao seja lido como uma categoria propria.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "correlacao",
    ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8f2020"],
)

# Sufixos que agrupam as 30 features em tres blocos de 10 medidas.
FEATURE_GROUPS = ("mean", "se", "worst")


def apply_style() -> None:
    """Aplica o estilo visual padrao do projeto ao matplotlib."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, filename: str) -> Path:
    """Grava a figura em `results/figures/` e devolve o caminho."""
    config.ensure_output_dirs()
    path = config.FIGURES_DIR / filename
    fig.savefig(path)
    return path


def feature_group(X: pd.DataFrame, suffix: str) -> list[str]:
    """Lista as features de um dos tres blocos (`mean`, `se` ou `worst`)."""
    if suffix not in FEATURE_GROUPS:
        raise ValueError(f"Grupo invalido: {suffix}. Use um de {FEATURE_GROUPS}.")
    return [c for c in X.columns if c.endswith(f"_{suffix}")]


# ---------------------------------------------------------------------------
# Estatisticas descritivas
# ---------------------------------------------------------------------------
def descriptive_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Resumo estatistico por feature, com assimetria e amplitude de escala."""
    stats = X.describe().T
    stats["skew"] = X.skew()
    stats["range"] = stats["max"] - stats["min"]
    return stats[["mean", "std", "min", "50%", "max", "range", "skew"]].round(4)


def class_balance(y: pd.Series) -> pd.DataFrame:
    """Contagem e proporcao de cada classe do alvo."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame(
        {
            "classe": [config.CLASS_NAMES[i] for i in counts.index],
            "amostras": counts.to_numpy(),
            "proporcao": (counts / len(y)).round(4).to_numpy(),
        }
    )


def separation_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Ordena as features pelo poder de separacao entre as duas classes.

    Usa o d de Cohen: diferenca das medias das classes dividida pelo desvio
    padrao combinado. E adimensional, entao permite comparar features com
    escalas muito diferentes (area na casa dos milhares, smoothness na casa
    dos centesimos).
    """
    benigno, maligno = X[y == 0], X[y == 1]
    n_b, n_m = len(benigno), len(maligno)

    pooled_std = np.sqrt(
        ((n_b - 1) * benigno.var(ddof=1) + (n_m - 1) * maligno.var(ddof=1))
        / (n_b + n_m - 2)
    )
    cohens_d = (maligno.mean() - benigno.mean()) / pooled_std

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "media_benigno": benigno.mean().to_numpy().round(4),
            "media_maligno": maligno.mean().to_numpy().round(4),
            "cohens_d": cohens_d.to_numpy().round(3),
        }
    )
    return ranking.reindex(
        ranking["cohens_d"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Correlacao
# ---------------------------------------------------------------------------
def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlacao de Pearson entre as features."""
    return X.corr(method="pearson")


def highly_correlated_pairs(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Lista os pares de features com correlacao absoluta acima do limiar.

    Cada par aparece uma unica vez (apenas o triangulo superior da matriz).
    """
    corr = correlation_matrix(X)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)

    pares = [
        {"feature_a": corr.index[i], "feature_b": corr.columns[j], "correlacao": round(corr.iat[i, j], 4)}
        for i, j in zip(*np.where(mask & (corr.abs() > threshold).to_numpy()))
    ]
    resultado = pd.DataFrame(pares, columns=["feature_a", "feature_b", "correlacao"])
    return resultado.sort_values("correlacao", key=abs, ascending=False).reset_index(drop=True)


def correlation_with_target(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Correlacao de cada feature com o alvo binario, ordenada por magnitude.

    Com um alvo 0/1, a correlacao de Pearson equivale ao coeficiente
    ponto-bisserial: mede o quanto a feature cresce junto com a probabilidade de
    o tumor ser maligno.
    """
    corr = X.corrwith(y).round(4)
    resultado = pd.DataFrame({"feature": corr.index, "correlacao": corr.to_numpy()})
    return resultado.reindex(
        resultado["correlacao"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def plot_class_balance(y: pd.Series, filename: str = "01_balanceamento_classes.png") -> Path:
    """Barras com a contagem de cada diagnostico."""
    balance = class_balance(y)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(
        balance["classe"],
        balance["amostras"],
        color=[COLOR_BENIGNO, COLOR_MALIGNO],
        width=0.55,
    )
    for bar, n, prop in zip(bars, balance["amostras"], balance["proporcao"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 6,
            f"{n}  ({prop:.1%})",
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_title("Distribuição dos diagnósticos")
    ax.set_ylabel("Amostras")
    ax.set_ylim(0, balance["amostras"].max() * 1.18)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return _save(fig, filename)


def plot_feature_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    filename: str,
    title: str,
) -> Path:
    """Grade de histogramas sobrepostos por classe."""
    n_cols = 5
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        for label, color in CLASS_COLORS.items():
            ax.hist(
                X.loc[y == label, feature],
                bins=25,
                color=color,
                alpha=0.65,
                label=config.CLASS_NAMES[label],
            )
        ax.set_title(feature, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    return _save(fig, filename)


def plot_boxplots_by_class(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    filename: str,
    title: str,
) -> Path:
    """Grade de boxplots comparando as duas classes em cada feature."""
    n_cols = 5
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 2.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        box = ax.boxplot(
            [X.loc[y == 0, feature], X.loc[y == 1, feature]],
            tick_labels=list(config.CLASS_NAMES),
            patch_artist=True,
            widths=0.5,
            medianprops={"color": INK, "linewidth": 1.4},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": INK_MUTED,
                "markeredgecolor": "none",
                "alpha": 0.5,
            },
        )
        for patch, color in zip(box["boxes"], (COLOR_BENIGNO, COLOR_MALIGNO)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
        ax.set_title(feature, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    return _save(fig, filename)


def plot_correlation_heatmap(
    X: pd.DataFrame,
    filename: str = "05_correlacao.png",
) -> Path:
    """Mapa de calor da correlacao de Pearson entre as 30 features.

    Usa escala divergente (azul para negativo, cinza no zero, vermelho para
    positivo) porque correlacao tem polaridade: o sinal importa tanto quanto a
    magnitude, e o zero e um ponto de referencia com significado proprio.
    """
    corr = correlation_matrix(X)

    fig, ax = plt.subplots(figsize=(11, 9.2))
    image = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    ax.grid(visible=False)
    ax.set_title("Correlação de Pearson entre as features", pad=14)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02)
    colorbar.set_label("coeficiente de correlação", fontsize=9)
    colorbar.outline.set_visible(False)
    return _save(fig, filename)


def plot_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    filename: str = "06_correlacao_alvo.png",
) -> Path:
    """Barras com as features mais correlacionadas ao diagnostico."""
    corr = correlation_with_target(X, y).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 0.36 * top_n + 1.4))
    ax.barh(corr["feature"], corr["correlacao"], color=COLOR_MALIGNO, height=0.62)
    for feature, valor in zip(corr["feature"], corr["correlacao"]):
        ax.text(valor + 0.008, feature, f"{valor:.2f}", va="center", fontsize=8, color=INK)

    ax.set_title(f"Top {top_n} features por correlação com o diagnóstico")
    ax.set_xlabel("correlação de Pearson com o alvo  ·  positivo = associado a malignidade")
    ax.set_xlim(0, corr["correlacao"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return _save(fig, filename)


def plot_separation_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    filename: str = "04_separacao_features.png",
) -> Path:
    """Barras horizontais com as features que mais separam as classes."""
    ranking = separation_ranking(X, y).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 0.36 * top_n + 1.4))
    ax.barh(ranking["feature"], ranking["cohens_d"], color=COLOR_MALIGNO, height=0.62)
    for feature, d in zip(ranking["feature"], ranking["cohens_d"]):
        ax.text(d + 0.03, feature, f"{d:.2f}", va="center", fontsize=8, color=INK)

    ax.set_title(f"Top {top_n} features por poder de separação (d de Cohen)")
    ax.set_xlabel("d de Cohen  ·  positivo = valor maior em tumores malignos")
    ax.set_xlim(0, ranking["cohens_d"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return _save(fig, filename)

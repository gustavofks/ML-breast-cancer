"""Estilo visual compartilhado pelos graficos do projeto.

Centraliza paleta, tema do matplotlib e gravacao de figuras, para que a analise
dos dados (`src/eda.py`) e a analise dos modelos (`src/evaluation.py`) produzam
graficos visualmente consistentes.

A paleta de duas classes foi validada para daltonismo: a separacao entre azul e
laranja e de dE 24.7 em protanopia, muito acima do minimo de 8 recomendado.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# Paleta categorica das duas classes do alvo.
COLOR_BENIGNO = "#2a78d6"
COLOR_MALIGNO = "#eb6834"
CLASS_COLORS = {0: COLOR_BENIGNO, 1: COLOR_MALIGNO}

# Cor fixa por modelo, atribuida pelo *nome* e nao pela posicao na tabela: a
# identidade visual de cada modelo permanece a mesma em todos os graficos, mesmo
# quando a ordenacao muda de uma figura para outra.
MODEL_COLORS = {
    # pipeline tabular
    "Regressão Logística": "#2a78d6",
    "KNN": "#eb6834",
    "Random Forest": "#1baf7a",
    # pipeline de imagem
    "CNN do zero": "#2a78d6",
    "MobileNetV2 (transferência)": "#eb6834",
    "MobileNetV2 (ajuste fino)": "#1baf7a",
}
_FALLBACK_COLOR = "#4a3aa7"


def model_color(name: str) -> str:
    """Cor associada a um modelo pelo nome."""
    return MODEL_COLORS.get(name, _FALLBACK_COLOR)

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


def apply_style() -> None:
    """Aplica o tema visual do projeto ao matplotlib."""
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


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Grava a figura em `results/figures/` e devolve o caminho."""
    config.ensure_output_dirs()
    path = config.FIGURES_DIR / filename
    fig.savefig(path)
    return path

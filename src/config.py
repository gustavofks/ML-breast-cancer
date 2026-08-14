"""Configuracoes centrais do projeto.

Fonte unica de caminhos, constantes e parametros de reprodutibilidade.
Nenhum outro modulo deve escrever caminhos literais.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
METRICS_FILE = RESULTS_DIR / "metrics.json"
REPORT_FILE = REPORTS_DIR / "index.html"

# ---------------------------------------------------------------------------
# Reprodutibilidade e particionamento
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# ---------------------------------------------------------------------------
ID_COLUMN = "id"
TARGET_COLUMN = "diagnosis"

# Maligno e a classe positiva: e o evento que se quer detectar, e o falso
# negativo (maligno classificado como benigno) e o erro de maior custo clinico.
TARGET_MAPPING = {"B": 0, "M": 1}
POSITIVE_LABEL = 1
CLASS_NAMES = ("Benigno", "Maligno")

# Usados na validacao do carregamento (ver src/data.py).
EXPECTED_N_SAMPLES = 569
EXPECTED_N_FEATURES = 30


def ensure_output_dirs() -> None:
    """Cria os diretorios de saida caso ainda nao existam."""
    for directory in (FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# Entrega extra: diagnostico por imagem
# ---------------------------------------------------------------------------
# A base de imagens nao e versionada (ver .gitignore). Espera-se a convencao
# de uma pasta por classe, adotada tanto pelo BUSI quanto pelas versoes em
# JPEG do CBIS-DDSM:
#
#     data/raw/images/<classe>/<arquivo>.png
#
IMAGES_DIR = RAW_DATA_DIR / "images"
VISION_METRICS_FILE = RESULTS_DIR / "metrics_vision.json"

IMAGE_SIZE = (224, 224)  # entrada padrao das redes pre-treinadas usadas
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.3  # metade vira validacao, metade vira teste


# ---------------------------------------------------------------------------
# Fase 2: algoritmo genetico
# ---------------------------------------------------------------------------
GENETIC_DIR = RESULTS_DIR / "genetic"

# Custo clinico assimetrico. A razao 10:1 traduz a politica descrita na secao
# 4.2 do relatorio da Fase 1: o falso negativo (tumor maligno classificado como
# benigno) e incomparavelmente mais grave que o falso positivo, que no fluxo
# clinico custa um exame adicional. A razao e decisao da instituicao de saude,
# entao vive aqui como parametro, nunca embutida no codigo do fitness.
CUSTO_FALSO_NEGATIVO = 10.0
CUSTO_FALSO_POSITIVO = 1.0

# Peso do termo de parcimonia. Serve de desempate: entre duas solucoes
# igualmente seguras, prefere a que exige menos medidas do patologista.
PENALIDADE_PARCIMONIA = 0.05

# Faixa util do limiar de decisao. Fora dela o classificador degenera em
# "tudo maligno" ou "tudo benigno", que nao e solucao candidata.
LIMIAR_MIN = 0.05
LIMIAR_MAX = 0.95

# Algoritmo genetico e estocastico: execucao unica nao e resultado. Toda
# configuracao roda com estas tres sementes, e o relatorio traz media e desvio.
GENETIC_SEEDS = (42, 43, 44)


def ensure_output_dirs() -> None:
    """Cria os diretorios de saida caso ainda nao existam."""
    for directory in (FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

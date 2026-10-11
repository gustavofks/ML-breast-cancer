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


# ---------------------------------------------------------------------------
# Fase 2: camada de modelo de linguagem
# ---------------------------------------------------------------------------
LLM_DIR = RESULTS_DIR / "llm"
LLM_CACHE_DIR = LLM_DIR / "cache"

# Modelo que gera os textos versionados. Decisao de 2026-10-11: Claude Haiku 5.5
# pela API, no lugar do modelo local do spec original. Gera texto melhor em
# portugues, dispensa instalar o Ollama, e o lote inteiro custa centavos. Toda a
# camada fala com um Protocol, entao trocar o modelo e mudar esta linha. O nome
# entra na chave do cache: texto gerado por outro modelo nunca e servido como se
# fosse deste.
LLM_MODELO = "claude-haiku-5-5"

# Alternativa gratuita e local, mantida funcionando.
OLLAMA_MODELO = "llama3.1:8b"
LLM_URL = "http://localhost:11434/api/generate"

# Temperatura zero e semente fixa. O texto que entra no relatorio precisa ser o
# mesmo a cada execucao, pela mesma regra que vale para results/metrics.json.
LLM_TEMPERATURA = 0.0
LLM_SEED = SEED

# Contrato de saida: o texto e para ser lido na tela, ao lado do grafico SHAP.
LLM_LIMITE_DE_PALAVRAS = 220
LLM_TIMEOUT = 300

# Custo do Claude, travado por construcao. O projeto e avaliado uma vez e o gasto
# tem de ficar muito abaixo de US$ 5, entao ha quatro travas: o modelo mais
# barato; esforco baixo, porque o pensamento adaptativo e cobrado como saida;
# teto de tokens por chamada; e um orcamento total que o provedor confere antes
# de cada chamada. O cache garante que texto ja gerado nunca e pago de novo.
CLAUDE_ESFORCO = "low"
CLAUDE_MAX_TOKENS = 2000  # pensamento + texto; 220 palavras sao ~400 tokens
LLM_ORCAMENTO_USD = 1.00

# Preco do Haiku 5.5 por milhao de tokens, para prompts de ate 100 mil tokens
# (os nossos tem menos de mil). Fonte: pagina oficial do modelo, 2026-10-11.
CLAUDE_PRECO_ENTRADA_USD = 0.10
CLAUDE_PRECO_SAIDA_USD = 0.50



# ---------------------------------------------------------------------------
# Fase 2: API
# ---------------------------------------------------------------------------
# Modelo servido. A escolha depende da avaliacao final dos bracos, entao vive
# num arquivo que o usuario grava depois de le-la, e nao no codigo. Sem o
# arquivo, a API serve a referencia da Fase 1 e declara isso em /model-info.
MODELO_SERVIDO_FILE = GENETIC_DIR / "modelo_servido.json"

# Desvios padrao a partir dos quais uma medida recebida dispara alerta. Medido
# sobre o treino: com 3, 13% dos casos legitimos alertariam, e alerta que
# dispara sempre vira ruido ignorado; com 5, 2,6%.
API_LIMIAR_DE_DESVIO = 5.0

# Variavel de ambiente com o commit de origem, preenchida no build do container.
API_VARIAVEL_DE_COMMIT = "APP_COMMIT"

# Publicacao no Hugging Face Spaces: modelos versionados do Dockerfile e do
# cartao do Space, e o diretorio onde o pacote e montado (fora do Git).
SPACE_MODELOS_DIR = PROJECT_ROOT / "deploy" / "spaces"
SPACE_PACOTE_DIR = PROJECT_ROOT / "build" / "space"

def ensure_output_dirs() -> None:
    """Cria os diretorios de saida caso ainda nao existam."""
    for directory in (FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

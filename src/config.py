"""Configuracoes centrais do projeto.

Fonte unica de caminhos, constantes e parametros de reprodutibilidade.
Nenhum outro modulo deve escrever caminhos literais.
"""

import os
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

# Processos que avaliam a populacao em paralelo; -1 usa todos os nucleos. Nao
# muda o resultado, so o tempo: a avaliacao de um individuo nao depende do
# gerador aleatorio da busca.
GENETIC_N_JOBS = -1


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

# Tamanho pedido no prompt, abaixo do limite que o verificador aplica. Com os
# dois iguais, o primeiro lote real passou do limite por 9 palavras: o modelo
# conta palavras de forma aproximada, e o contrato precisa de folga.
LLM_ALVO_DE_PALAVRAS = 180

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

# Eventos de log guardados em memoria para a vitrine mostrar na tela. Duzentos
# cobrem varios minutos de demonstracao e custam poucos kilobytes.
API_EVENTOS_EM_MEMORIA = 200

# Publicacao no Hugging Face Spaces: modelos versionados do Dockerfile e do
# cartao do Space, e o diretorio onde o pacote e montado (fora do Git).
SPACE_MODELOS_DIR = PROJECT_ROOT / "deploy" / "spaces"
SPACE_PACOTE_DIR = PROJECT_ROOT / "build" / "space"

# Ativos da vitrine (molde, estilo, script e o Chart.js versionado), embutidos
# no HTML na montagem: a pagina nao carrega nada de fora.
VITRINE_DIR = PROJECT_ROOT / "src" / "ui" / "vitrine"

# ---------------------------------------------------------------------------
# Fase 3: clinica ficticia e medicoes
# ---------------------------------------------------------------------------
# "Clinica", e nao "hospital": HOSPITAL_* ja nomeia a maquete 3D da
# apresentacao, e os dois convivem quando as branches se juntarem.
CLINICA_NOME = "Hospital Escola Aurora"
CLINICA_DIR = DATA_DIR / "clinica"
CLINICA_BRUTO_DIR = CLINICA_DIR / "bruto"  # fora do Git
CLINICA_PRONTUARIOS_FILE = CLINICA_DIR / "prontuarios.jsonl"
CLINICA_GABARITO_FILE = CLINICA_DIR / "gabarito_paaf.json"
CLINICA_BANCO_FILE = CLINICA_DIR / "prontuarios.sqlite"  # reconstruido, fora do Git

# 114 pacientes recebem os casos de teste da Fase 2 como PAAF; as demais tem so
# imagem, como na rotina de uma clinica de mama.
CLINICA_N_PACIENTES = 150

# "Hoje" da clinica, fixo: pendencia depende da data de hoje, e o banco
# precisa ser o mesmo a cada execucao, pela mesma regra do metrics.json.
CLINICA_HOJE = "2026-11-01"

# Exame solicitado sem resultado ha mais que isto vira pendencia.
CLINICA_PRAZO_PENDENCIA_DIAS = 30

FASE3_DIR = RESULTS_DIR / "fase3"
FASE3_MEDICOES_DIR = FASE3_DIR / "medicoes"

# Modelos escolhidos na semana 1 (results/fase3/medicoes/).
FASE3_MODELO_BASE = "Qwen/Qwen3.5-4B"
FASE3_PROFESSOR = "Qwen/Qwen3.5-9B"
FASE3_ORIGEM_PROFESSOR = f"professor:{FASE3_PROFESSOR}"

# Servidor do llama.cpp no WSL, visto do Windows e do proprio WSL.
FASE3_PROFESSOR_URL = "http://localhost:8091"

# Textos da clinica: protocolos em markdown, saida bruta do professor, conjuntos.
CLINICA_PROTOCOLOS_DIR = CLINICA_DIR / "protocolos"
CLINICA_GERADO_DIR = CLINICA_DIR / "gerado"
CLINICA_CONJUNTOS_DIR = CLINICA_DIR / "conjuntos"
FASE3_CACHE_PROFESSOR_DIR = CLINICA_GERADO_DIR / "cache"

# Fontes do INCA: fora do Git (direitos reservados no repositorio do INCA).
FONTES_DIR = DATA_DIR / "fontes"

# Volumes da geracao. Perguntas por protocolo: o bastante para o modelo ver o
# formato muitas vezes, pouco o bastante para o professor gerar em uma noite.
FASE3_PERGUNTAS_TREINO_POR_PROTOCOLO = 30
FASE3_PERGUNTAS_TESTE_POR_PROTOCOLO = 8
FASE3_N_RECUSAS = 60
FASE3_N_FORA_DO_ESCOPO = 40
FASE3_N_GERAIS = 150
FASE3_N_NAO_ENCONTRADO = 60

# Fracao do treino separada para validacao, e fracao dos exemplos sem
# protocolo (recusas, gerais...) que vai para o teste.
FASE3_FRACAO_VALIDACAO = 0.1
FASE3_FRACAO_TESTE_SEM_PROTOCOLO = 0.2

# Semelhanca a partir da qual duas perguntas contam como a mesma. 0,9 pega
# reescritas leves ("Qual a idade..." / "Qual e a idade...") sem fundir perguntas
# diferentes sobre o mesmo tema.
FASE3_LIMIAR_DUPLICATA = 0.9

# Frase apontada pelo professor como sem base so sai se menos da metade das
# palavras dela estiver na fonte. Na primeira geracao, as 25 frases apontadas
# tinham entre 87% e 100% das palavras na fonte: o verificador sozinho errava.
FASE3_SUPORTE_MINIMO = 0.5

# Receita do fine-tuning (semana 3). LoRA r=16 em todas as camadas lineares,
# como no teste da semana 1; taxa e epocas conservadoras porque o teste da GPU
# mostrou o modelo esquecendo o resto depois de decorar 40 codigos. A epoca
# servida e a de menor perda de validacao, nao a ultima.
FASE3_LORA_R = 16
FASE3_LORA_ALPHA = 32
FASE3_LORA_DROPOUT = 0.05
FASE3_TREINO_TAXA = 2e-4
FASE3_TREINO_EPOCAS = 3
FASE3_TREINO_ACUMULACAO = 8  # lote efetivo de 8 exemplos, um por vez na placa
FASE3_TREINO_MAX_TOKENS = 1024  # maior exemplo medido: 989 tokens

FASE3_TREINO_DIR = FASE3_DIR / "treino"

# PROT-13 v2: prazo de pendencia depois da "atualizacao" do protocolo.
CLINICA_PRAZO_PENDENCIA_DIAS_V2 = 21

# Assistente (semana 4). Os dois servidores do llama.cpp sobem com
# scripts/wsl/servir_assistente.sh; no container, a variavel de ambiente aponta
# para host.docker.internal.
FASE3_AJUSTADO_URL = os.environ.get("FASE3_AJUSTADO_URL", "http://localhost:8092")
FASE3_BASE_URL = os.environ.get("FASE3_BASE_URL", "http://localhost:8093")
FASE3_RESPOSTA_MAX_TOKENS = 600  # a maior resposta do ajustado na semana 3 usou bem menos
# Texto fixo anexado pela cadeia, e nao pedido ao modelo: na semana 3 nenhuma
# resposta do ajustado fechou com a decisao do medico.
FASE3_FECHO = "A decisão final é do médico responsável."
FASE3_AVISO_GERAL = ("Nenhum protocolo do hospital trata desta pergunta: resposta de conhecimento "
                     "geral do modelo, a conferir.")
FASE3_CADEIA_DIR = FASE3_DIR / "cadeia"

# Busca (semana 4), escolhida pela regra de results/fase3/medicoes/regra_busca.json
# com a medicao de busca.json: o BM25 nos 4 melhores trechos acertou 0,885 das
# perguntas de protocolo, contra 0,769 do protocolo inteiro e 0,870 do e5-small.
# Abaixo do limiar, a pergunta e tratada como geral e vai sem trechos.
FASE3_BUSCA_MODO = "trechos"
FASE3_BUSCA_K = 4
FASE3_BUSCA_LIMIAR = 0.8954


def ensure_output_dirs() -> None:
    """Cria os diretorios de saida caso ainda nao existam."""
    for directory in (FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

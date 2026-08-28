# Pipeline do Tech Challenge Fase 1 em um container reprodutivel.
#
# O container executa a analise, nao a serve: `docker run` roda
# `scripts/run_wisconsin.py` de ponta a ponta e grava figuras, metricas e o
# relatorio HTML em `/app/results`. Monte esse caminho para receber os
# arquivos na maquina.
#
#   docker build -t ml-breast-cancer .
#   docker run --rm -v "$(pwd)/results:/app/results" ml-breast-cancer
#
# A entrega extra com imagens vive em um estagio separado, porque o TensorFlow
# multiplica o tamanho da imagem e a base BUSI nao e versionada:
#
#   docker build --target vision -t ml-breast-cancer:vision .
#   docker run --rm -v "$(pwd)/results:/app/results" \
#              -v "$(pwd)/data/raw/images:/app/data/raw/images" \
#              ml-breast-cancer:vision

# ---------------------------------------------------------------------------
# Base comum: dependencias do pipeline tabular, codigo e usuario sem
# privilegios. Nao e o alvo padrao — quem constroi sem `--target` recebe o
# estagio `tabular`, declarado por ultimo.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# PYTHONUNBUFFERED: o pipeline imprime o progresso etapa a etapa, e sem isso a
# saida so apareceria no fim, em bloco.
# MPLCONFIGDIR: o matplotlib precisa de um diretorio de cache com permissao de
# escrita; sem ele o usuario sem privilegios recebe um aviso a cada execucao.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# As dependencias vem antes do codigo de proposito: esta camada so e
# reconstruida quando `requirements.txt` muda, e nao a cada edicao em `src/`.
# O arquivo e o mesmo do README — o container instala exatamente o que uma
# instalacao local instalaria.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY conftest.py ./
COPY data/raw/data.csv ./data/raw/data.csv

# Usuario sem privilegios. O uid 1000 e o primeiro usuario comum em Linux, o
# que faz os arquivos gravados no volume montado pertencerem a quem rodou o
# container, em vez de a root.
RUN useradd --create-home --uid 1000 analise \
    && mkdir -p results/figures results/reports \
    && chown -R analise:analise /app

USER analise

# Carrega e valida os dados, gera as figuras da analise exploratoria, compara
# os modelos por validacao cruzada, avalia no teste, calcula as explicacoes e
# monta o relatorio HTML. Idempotente: rodar de novo regenera tudo.
CMD ["python", "scripts/run_wisconsin.py"]


# ---------------------------------------------------------------------------
# Estagio opcional: entrega extra com diagnostico por imagem.
#
# Separado porque o TensorFlow sozinho pesa mais que todo o resto da imagem, e
# porque a base BUSI (780 imagens, 256 MB) nao e versionada — sem montar
# `data/raw/images`, este estagio nao tem o que treinar.
# ---------------------------------------------------------------------------
FROM base AS vision

USER root
COPY requirements-vision.txt ./
RUN pip install --no-cache-dir -r requirements-vision.txt
USER analise

CMD ["python", "scripts/run_vision.py"]


# ---------------------------------------------------------------------------
# Alvo padrao. Precisa ser o ultimo estagio do arquivo: sem `--target`, o
# Docker constroi o estagio final — e sem esta linha o padrao seria `vision`,
# com o TensorFlow inteiro a reboque.
# ---------------------------------------------------------------------------
FROM base AS tabular

# Imagem publicada no Hugging Face Spaces: API e painel numa porta so.
#
# Montada por `scripts/montar_space.py` junto com o pacote do Space. Existe
# separada do Dockerfile do projeto porque o Spaces constroi sem `--target`, e
# la o estagio padrao e o `tabular`, que roda a analise em vez de servir. Espelha
# os estagios `base` e `api`, instalando os mesmos arquivos de requisitos: as
# versoes nao sao repetidas aqui, e nao podem divergir.
#
# O container nao fala com a API do Claude nem com a internet. Le o modelo
# servido e o cache de textos em `results/`, gerados na maquina de
# desenvolvimento e copiados para o pacote.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY src/ ./src/
COPY data/ ./data/
COPY results/ ./results/

# O Spaces executa o container com o uid 1000.
RUN useradd --create-home --uid 1000 analise \
    && chown -R analise:analise /app
USER analise

# Preenchido pelo script de montagem com o commit de origem.
ENV APP_COMMIT=__APP_COMMIT__

# O Spaces encaminha o trafego para a porta 7860.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.ui.app:criar_app_com_painel --factory --host 0.0.0.0 --port ${PORT}"]

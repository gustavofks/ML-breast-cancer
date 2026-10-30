"""Ponto de entrada do Space com o SDK Gradio.

O SDK Gradio executa `python app.py` e encaminha o trafego para a porta 7860.
Aqui sobe o mesmo servidor do container: a API e o painel, num processo so, com
o painel em `/painel/` e a raiz redirecionando para ele.

Copiado para a raiz do pacote por `scripts/montar_space.py`.
"""

import os

import uvicorn

# O painel fala com a API pelo endereco local do processo, lido de PORT; o SDK
# Gradio informa a porta em GRADIO_SERVER_PORT.
os.environ.setdefault("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))
# Preenchido pelo script de montagem com o commit de origem.
os.environ.setdefault("APP_COMMIT", "__APP_COMMIT__")

from src.ui.app import criar_app_com_painel  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(criar_app_com_painel(), host="0.0.0.0", port=int(os.environ["PORT"]))

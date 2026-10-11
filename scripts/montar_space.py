"""Monta o pacote publicado no Hugging Face Spaces, em `build/space/`.

Uso, a partir da raiz do projeto:

    python scripts/montar_space.py

O Space e um repositorio Git separado, e o que vai para ele e so o que o
container precisa para servir: codigo de `src/`, a base, os requisitos, o modelo
servido e o cache de textos. Ficam de fora os testes, os notebooks, os
documentos e o SDK da API do Claude — o container publicado nunca fala com ela, e
levar o SDK ou um arquivo de ambiente seria convite a publicar uma chave.

O pacote e refeito do zero a cada execucao: arquivo que saiu do projeto nao pode
continuar no Space por inercia.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

LEIAME_DE_RESULTADOS = """# results/

Artefatos lidos pelo container: `genetic/modelo_servido.json` (o modelo servido)
e `llm/cache/` (os textos gerados na maquina de desenvolvimento).
"""


def commit_de_origem() -> str:
    """Commit curto do projeto.

    Sem Git (o pacote estatico e montado dentro do container, que nao leva o
    `.git`), usa o `APP_COMMIT` do ambiente; sem os dois, `desconhecido`.
    """
    reserva = os.environ.get(config.API_VARIAVEL_DE_COMMIT, "desconhecido")
    try:
        saida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return reserva
    return saida.stdout.strip() or reserva


TITULO = "Apoio ao diagnóstico de câncer de mama"
# Os mesmos valores preenchidos no formulario de criacao do Space: o README do
# pacote substitui o que o Spaces criou, e sem eles a descricao e a licenca
# escolhidas sumiriam no primeiro envio.
DESCRICAO_CURTA = "Apoio ao diagnóstico de câncer de mama com IA explicável"
LICENCA = "mit"


def versao_do_gradio() -> str:
    """A versao fixada em `requirements-api.txt`, fonte unica do pin."""
    texto = (config.PROJECT_ROOT / "requirements-api.txt").read_text(encoding="utf-8")
    return re.search(r"^gradio==(\S+)$", texto, re.M).group(1)


def cabecalho(sdk: str) -> str:
    """Bloco YAML do cartao do Space, que diz ao Spaces como executar o pacote."""
    linhas = [
        f"title: {TITULO}",
        f"short_description: {DESCRICAO_CURTA}",
        f"license: {LICENCA}",
        "colorFrom: blue",
        "colorTo: yellow",
        f"sdk: {sdk}",
    ]
    if sdk == "docker":
        linhas.append("app_port: 7860")
    elif sdk == "static":
        linhas.append("app_file: index.html")
    else:
        # O SDK Gradio instala o Gradio por esta versao, e nao pelo
        # requirements.txt: ela tem de ser a mesma do pin, senao o Space roda
        # uma versao que os testes nunca viram.
        linhas += [
            f"sdk_version: {versao_do_gradio()}",
            'python_version: "3.11"',
            "app_file: app.py",
        ]
    linhas.append("pinned: false")
    return "---\n" + "\n".join(linhas) + "\n---\n\n"


def montar(
    destino: Path = config.SPACE_PACOTE_DIR,
    *,
    resultados: Path = config.RESULTS_DIR,
    sdk: str = "static",
) -> list[str]:
    """Monta o pacote e devolve os avisos sobre o que falta para publicar.

    Tres formatos. O **estatico**, o unico gratuito para conta pessoal desde que
    o Hugging Face passou a cobrar por Spaces que executam codigo, publica so a
    pagina gerada por `src.ui.estatico`. O SDK Gradio executa `app.py` e so
    instala `requirements.txt`, entao os requisitos da Fase 1 e os da API viram
    um arquivo so. O SDK Docker constroi o Dockerfile do pacote. Gradio e Docker
    sobem o mesmo servidor, API e painel.
    """
    if sdk not in ("static", "gradio", "docker"):
        raise ValueError(f"sdk desconhecido: {sdk!r}")
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    commit = commit_de_origem()

    if sdk == "static":
        (destino / "README.md").write_text(
            cabecalho(sdk) + CORPO_ESTATICO, encoding="utf-8"
        )
        return _montar_estatico(destino, resultados, commit)

    corpo = (config.SPACE_MODELOS_DIR / "README.md").read_text(encoding="utf-8")
    (destino / "README.md").write_text(cabecalho(sdk) + corpo, encoding="utf-8")

    if sdk == "docker":
        dockerfile = (config.SPACE_MODELOS_DIR / "Dockerfile").read_text(encoding="utf-8")
        (destino / "Dockerfile").write_text(
            dockerfile.replace("__APP_COMMIT__", commit), encoding="utf-8"
        )
        for requisitos in ("requirements.txt", "requirements-api.txt"):
            shutil.copy2(config.PROJECT_ROOT / requisitos, destino / requisitos)
    else:
        app = (config.SPACE_MODELOS_DIR / "app.py").read_text(encoding="utf-8")
        (destino / "app.py").write_text(app.replace("__APP_COMMIT__", commit), encoding="utf-8")
        juntos = "\n".join(
            (config.PROJECT_ROOT / nome).read_text(encoding="utf-8")
            for nome in ("requirements.txt", "requirements-api.txt")
        )
        (destino / "requirements.txt").write_text(juntos, encoding="utf-8")

    shutil.copytree(
        config.PROJECT_ROOT / "src",
        destino / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    base = destino / "data" / "raw"
    base.mkdir(parents=True)
    shutil.copy2(config.RAW_DATA_FILE, base / config.RAW_DATA_FILE.name)

    # `results/` sempre existe no pacote, mesmo vazio: o Dockerfile o copia, e o
    # Git nao versiona diretorio vazio.
    (destino / "results").mkdir()
    (destino / "results" / "LEIAME.md").write_text(LEIAME_DE_RESULTADOS, encoding="utf-8")

    avisos = []
    modelo = resultados / "genetic" / "modelo_servido.json"
    if modelo.exists():
        (destino / "results" / "genetic").mkdir()
        shutil.copy2(modelo, destino / "results" / "genetic" / modelo.name)
    else:
        avisos.append(
            "sem results/genetic/modelo_servido.json: o Space vai servir a "
            "referencia da Fase 1"
        )

    entradas = sorted((resultados / "llm" / "cache").glob("*.json"))
    if entradas:
        cache = destino / "results" / "llm" / "cache"
        cache.mkdir(parents=True)
        for entrada in entradas:
            shutil.copy2(entrada, cache / entrada.name)
    else:
        avisos.append(
            "cache de textos vazio: todo caso aparece como texto indisponivel; "
            "rode scripts/run_llm.py --casos todos antes de publicar"
        )
    return avisos


CORPO_ESTATICO = """# Apoio ao diagnóstico de câncer de mama

Tech Challenge da pós-graduação em IA da FIAP, Fase 2. Escolha um caso do
conjunto de teste da base Breast Cancer Wisconsin e veja a probabilidade de
malignidade, o limiar de decisão, as medidas que mais pesaram e uma explicação em
linguagem natural.

**Sistema de apoio à decisão.** A previsão e a explicação ajudam a leitura do
caso, mas o diagnóstico final é sempre do médico responsável.

Esta é a versão estática do painel: os números e os textos foram gerados pela
API do projeto e congelados na página. A API completa, com previsão para medidas
novas, log estruturado e métricas, roda no container do repositório.
"""


def _avisos(resultados: Path) -> list[str]:
    """O que falta em `results/` para o Space mostrar o modelo e os textos."""
    avisos = []
    if not (resultados / "genetic" / "modelo_servido.json").exists():
        avisos.append(
            "sem results/genetic/modelo_servido.json: o Space vai servir a "
            "referencia da Fase 1"
        )
    if not any((resultados / "llm" / "cache").glob("*.json")):
        avisos.append(
            "cache de textos vazio: todo caso aparece como texto indisponivel; "
            "rode scripts/run_llm.py --casos todos antes de publicar"
        )
    return avisos


def _montar_estatico(destino: Path, resultados: Path, commit: str) -> list[str]:
    """Gera `index.html` consultando a API em processo, com o modelo e o cache dados.

    Os imports ficam aqui dentro: so este formato precisa da FastAPI na maquina
    que monta o pacote.
    """

    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.api.servico import Servico, carregar_especificacao
    from src.ui import estatico

    modelo = Servico.construir(
        carregar_especificacao(resultados / "genetic" / "modelo_servido.json")
    )
    api = TestClient(api_main.criar_app(modelo, diretorio_cache=resultados / "llm" / "cache"))

    anterior = os.environ.get(config.API_VARIAVEL_DE_COMMIT)
    os.environ[config.API_VARIAVEL_DE_COMMIT] = commit
    try:
        dados = estatico.dados_dos_casos(api)
    finally:
        if anterior is None:
            os.environ.pop(config.API_VARIAVEL_DE_COMMIT, None)
        else:
            os.environ[config.API_VARIAVEL_DE_COMMIT] = anterior

    (destino / "index.html").write_text(estatico.pagina(dados), encoding="utf-8")
    return _avisos(resultados)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pacote do Hugging Face Spaces.")
    parser.add_argument("--destino", type=Path, default=config.SPACE_PACOTE_DIR)
    parser.add_argument(
        "--sdk",
        choices=("static", "gradio", "docker"),
        default="static",
        help="static (padrao, o unico gratuito para conta pessoal), gradio ou docker",
    )
    argumentos = parser.parse_args()

    avisos = montar(argumentos.destino, sdk=argumentos.sdk)
    print(f"Pacote montado em {argumentos.destino}")
    for aviso in avisos:
        print(f"  aviso: {aviso}")


if __name__ == "__main__":
    main()

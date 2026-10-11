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
    """Commit curto do projeto, ou `desconhecido` fora de um repositorio Git."""
    try:
        saida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"
    return saida.stdout.strip() or "desconhecido"


TITULO = "Apoio ao diagnóstico de câncer de mama"


def versao_do_gradio() -> str:
    """A versao fixada em `requirements-api.txt`, fonte unica do pin."""
    texto = (config.PROJECT_ROOT / "requirements-api.txt").read_text(encoding="utf-8")
    return re.search(r"^gradio==(\S+)$", texto, re.M).group(1)


def cabecalho(sdk: str) -> str:
    """Bloco YAML do cartao do Space, que diz ao Spaces como executar o pacote."""
    linhas = [f"title: {TITULO}", "colorFrom: blue", "colorTo: yellow", f"sdk: {sdk}"]
    if sdk == "docker":
        linhas.append("app_port: 7860")
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
    sdk: str = "gradio",
) -> list[str]:
    """Monta o pacote e devolve os avisos sobre o que falta para publicar.

    Dois formatos. O SDK Gradio, gratuito, executa `app.py` e so instala
    `requirements.txt`, entao os requisitos da Fase 1 e os da API viram um
    arquivo so. O SDK Docker, que apareceu como pago na criacao do Space,
    constroi o Dockerfile do pacote. Os dois sobem o mesmo servidor.
    """
    if sdk not in ("gradio", "docker"):
        raise ValueError(f"sdk desconhecido: {sdk!r}")
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    commit = commit_de_origem()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pacote do Hugging Face Spaces.")
    parser.add_argument("--destino", type=Path, default=config.SPACE_PACOTE_DIR)
    parser.add_argument(
        "--sdk",
        choices=("gradio", "docker"),
        default="gradio",
        help="gradio (gratuito, padrao) ou docker",
    )
    argumentos = parser.parse_args()

    avisos = montar(argumentos.destino, sdk=argumentos.sdk)
    print(f"Pacote montado em {argumentos.destino}")
    for aviso in avisos:
        print(f"  aviso: {aviso}")


if __name__ == "__main__":
    main()

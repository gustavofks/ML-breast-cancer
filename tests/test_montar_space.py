"""Testes do pacote publicado no Hugging Face Spaces."""

import importlib.util

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "montar_space.py"
    spec = importlib.util.spec_from_file_location("montar_space", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture
def resultados(tmp_path):
    """Um `results/` com modelo servido e duas entradas de cache."""
    raiz = tmp_path / "results"
    (raiz / "genetic").mkdir(parents=True)
    (raiz / "genetic" / "modelo_servido.json").write_text("{}", encoding="utf-8")
    (raiz / "llm" / "cache").mkdir(parents=True)
    for nome in ("a", "b"):
        (raiz / "llm" / "cache" / f"{nome}.json").write_text("{}", encoding="utf-8")
    return raiz


def test_pacote_traz_o_que_o_container_precisa_para_servir(script, resultados, tmp_path):
    destino = tmp_path / "space"

    script.montar(destino, resultados=resultados)

    for caminho in (
        "Dockerfile",
        "README.md",
        "requirements.txt",
        "requirements-api.txt",
        "src/api/main.py",
        "src/ui/app.py",
        "data/raw/data.csv",
        "results/genetic/modelo_servido.json",
        "results/llm/cache/a.json",
    ):
        assert (destino / caminho).exists(), caminho


def test_cartao_do_space_declara_docker_e_a_porta_7860(script, resultados, tmp_path):
    """Sem `app_port`, o Spaces procura a porta 7860 por padrao; declarar evita surpresa."""
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados)

    cartao = (destino / "README.md").read_text(encoding="utf-8")
    cabecalho = cartao.split("---")[1]

    assert "sdk: docker" in cabecalho
    assert "app_port: 7860" in cabecalho


def test_dockerfile_do_space_serve_na_porta_7860_sem_privilegio(script, resultados, tmp_path):
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados)

    dockerfile = (destino / "Dockerfile").read_text(encoding="utf-8")

    assert "ENV PORT=7860" in dockerfile
    assert "--uid 1000" in dockerfile
    assert "criar_app_com_painel" in dockerfile


def test_pacote_nao_leva_testes_chaves_nem_o_sdk_da_api_paga(script, resultados, tmp_path):
    """O container publicado nunca fala com a API do Claude: le o cache e so.

    Levar o SDK ou um arquivo de ambiente seria convite a publicar uma chave.
    """
    (resultados.parent / ".env").write_text("ANTHROPIC_API_KEY=x", encoding="utf-8")
    destino = tmp_path / "space"

    script.montar(destino, resultados=resultados)

    nomes = {p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file()}
    assert not any(n.startswith("tests/") for n in nomes)
    assert "requirements-llm.txt" not in nomes
    assert not any(n.endswith(".env") for n in nomes)
    assert not any("__pycache__" in n for n in nomes)


def test_pacote_sem_texto_gerado_ou_sem_modelo_escolhido_avisa(script, tmp_path):
    """Publicar sem cache mostra "texto indisponivel" em todo caso; o script diz isso."""
    vazio = tmp_path / "results"
    vazio.mkdir()

    avisos = script.montar(tmp_path / "space", resultados=vazio)

    assert any("modelo_servido.json" in aviso for aviso in avisos)
    assert any("cache" in aviso for aviso in avisos)


def test_montar_de_novo_substitui_o_pacote_anterior(script, resultados, tmp_path):
    """Arquivo que saiu do projeto nao pode continuar no pacote por inercia."""
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados)
    (destino / "sobra.txt").write_text("velho", encoding="utf-8")

    script.montar(destino, resultados=resultados)

    assert not (destino / "sobra.txt").exists()

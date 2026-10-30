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

    script.montar(destino, resultados=resultados, sdk="docker")

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
    script.montar(destino, resultados=resultados, sdk="docker")

    cartao = (destino / "README.md").read_text(encoding="utf-8")
    cabecalho = cartao.split("---")[1]

    assert "sdk: docker" in cabecalho
    assert "app_port: 7860" in cabecalho


def test_dockerfile_do_space_serve_na_porta_7860_sem_privilegio(script, resultados, tmp_path):
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados, sdk="docker")

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

    script.montar(destino, resultados=resultados, sdk="gradio")

    nomes = {p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file()}
    assert not any(n.startswith("tests/") for n in nomes)
    assert "requirements-llm.txt" not in nomes
    assert not any(n.endswith(".env") for n in nomes)
    assert not any("__pycache__" in n for n in nomes)


def test_pacote_sem_texto_gerado_ou_sem_modelo_escolhido_avisa(script, tmp_path):
    """Publicar sem cache mostra "texto indisponivel" em todo caso; o script diz isso."""
    vazio = tmp_path / "results"
    vazio.mkdir()

    avisos = script.montar(tmp_path / "space", resultados=vazio, sdk="gradio")

    assert any("modelo_servido.json" in aviso for aviso in avisos)
    assert any("cache" in aviso for aviso in avisos)


def test_montar_de_novo_substitui_o_pacote_anterior(script, resultados, tmp_path):
    """Arquivo que saiu do projeto nao pode continuar no pacote por inercia."""
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados, sdk="gradio")
    (destino / "sobra.txt").write_text("velho", encoding="utf-8")

    script.montar(destino, resultados=resultados, sdk="gradio")

    assert not (destino / "sobra.txt").exists()


# ---------------------------------------------------------------------------
# Space com o SDK Gradio, gratuito
# ---------------------------------------------------------------------------
def test_pacote_gradio_tem_app_py_e_cartao_do_sdk_gradio(script, resultados, tmp_path):
    """O SDK Docker apareceu como pago na criacao do Space; o Gradio e gratuito."""
    destino = tmp_path / "space"

    script.montar(destino, resultados=resultados, sdk="gradio")

    assert (destino / "app.py").exists()
    assert not (destino / "Dockerfile").exists()
    cabecalho = (destino / "README.md").read_text(encoding="utf-8").split("---")[1]
    assert "sdk: gradio" in cabecalho
    assert "app_file: app.py" in cabecalho
    assert 'python_version: "3.11"' in cabecalho


def test_versao_do_gradio_no_cartao_e_a_mesma_fixada_nos_requisitos(script, resultados, tmp_path):
    """O Spaces instala o Gradio pelo `sdk_version`; divergir do pin quebraria a paridade."""
    import re

    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados, sdk="gradio")

    cabecalho = (destino / "README.md").read_text(encoding="utf-8").split("---")[1]
    fixada = re.search(
        r"^gradio==(\S+)$",
        (config.PROJECT_ROOT / "requirements-api.txt").read_text(encoding="utf-8"),
        re.M,
    ).group(1)
    assert f"sdk_version: {fixada}" in cabecalho


def test_pacote_gradio_junta_os_requisitos_num_arquivo_so(script, resultados, tmp_path):
    """O SDK Gradio so instala `requirements.txt`; a API precisa da FastAPI dele."""
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados, sdk="gradio")

    requisitos = (destino / "requirements.txt").read_text(encoding="utf-8")

    assert "fastapi==" in requisitos
    assert "scikit-learn==" in requisitos
    assert not (destino / "requirements-api.txt").exists()


def test_app_py_serve_api_e_painel_na_porta_do_spaces(script, resultados, tmp_path):
    destino = tmp_path / "space"
    script.montar(destino, resultados=resultados, sdk="gradio")

    app = (destino / "app.py").read_text(encoding="utf-8")

    assert "criar_app_com_painel" in app
    assert "7860" in app


# ---------------------------------------------------------------------------
# Space estatico: o unico gratuito para conta pessoal
# ---------------------------------------------------------------------------
def test_pacote_estatico_tem_so_a_pagina_e_o_cartao(script, tmp_path):
    """Space estatico serve arquivos; codigo, base e requisitos nao tem funcao la."""
    pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")
    destino = tmp_path / "space"

    script.montar(destino, sdk="static")

    nomes = sorted(p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file())
    assert nomes == ["README.md", "index.html"]
    cabecalho = (destino / "README.md").read_text(encoding="utf-8").split("---")[1]
    assert "sdk: static" in cabecalho
    assert "app_file: index.html" in cabecalho
    # O Spaces limita a descricao curta a 60 caracteres.
    descricao = next(l for l in cabecalho.splitlines() if l.startswith("short_description:"))
    assert len(descricao.split(":", 1)[1].strip()) <= 60
    assert "license: mit" in cabecalho


def test_pagina_estatica_usa_o_modelo_e_o_cache_dos_resultados(script, tmp_path):
    """Os 114 casos com texto: a pagina vem do modelo servido e do cache gerado."""
    pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")
    destino = tmp_path / "space"

    avisos = script.montar(destino, sdk="static")

    html = (destino / "index.html").read_text(encoding="utf-8")
    assert avisos == []
    assert html.count('"texto_motivo": null') == 114


def test_sem_git_o_commit_vem_do_ambiente(script, monkeypatch):
    """O pacote estatico e montado no container, que nao leva o `.git`."""

    def sem_git(*args, **kwargs):
        raise OSError("git ausente")

    monkeypatch.setattr(script.subprocess, "run", sem_git)
    monkeypatch.setenv("APP_COMMIT", "abc1234")

    assert script.commit_de_origem() == "abc1234"

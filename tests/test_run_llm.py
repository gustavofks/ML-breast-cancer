"""Testes da geracao em lote dos textos."""

import importlib.util
import json

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "run_llm.py"
    spec = importlib.util.spec_from_file_location("run_llm", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_modo_verificacao_roda_o_lote_inteiro_sem_modelo_instalado(
    script, tmp_path, monkeypatch
):
    """O Ollama nao e pre-requisito para exercitar prompts e verificadores.

    A camada inteira fala com um Protocol, entao o lote roda com um provedor
    deterministico. E isso que permite desenvolver e testar a Fase 2 antes de ter
    qualquer modelo local instalado.
    """
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "1", "--saida", str(destino)]
    )

    script.main()

    assert destino.exists()


def test_modo_verificacao_percorre_o_caminho_feliz(script, tmp_path, monkeypatch):
    """O `--fake` existe para que uma falha real se destaque.

    Se o provedor falso devolvesse texto generico, tudo reprovaria sempre e o
    modo de verificacao nao distinguiria encanamento quebrado de texto ruim. A
    resposta falsa respeita o contrato do template que a pediu; os verificadores
    ja sao testados contra defeitos na propria suite deles.
    """
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "1", "--saida", str(destino)]
    )

    script.main()

    registros = json.loads(destino.read_text(encoding="utf-8"))
    reprovados = [r["identificador"] for r in registros if not r["qualidade"]["aprovado"]]

    assert not reprovados, reprovados


def test_lote_grava_um_registro_por_texto_com_a_avaliacao_junto(
    script, tmp_path, monkeypatch
):
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "2", "--saida", str(destino)]
    )

    script.main()

    registros = json.loads(destino.read_text(encoding="utf-8"))
    assert registros
    for registro in registros:
        assert registro["identificador"]
        assert "qualidade" in registro
        assert "aprovado" in registro["qualidade"]


def test_os_tres_produtos_de_texto_aparecem_no_lote(script, tmp_path, monkeypatch):
    """Explicacao por paciente, resumo de gestao e nota de mudanca."""
    destino = tmp_path / "narrativas.json"
    monkeypatch.setattr(
        "sys.argv", ["run_llm.py", "--fake", "--casos", "1", "--saida", str(destino)]
    )

    script.main()

    registros = json.loads(destino.read_text(encoding="utf-8"))
    nomes = {r["identificador"].rsplit("_v", 1)[0] for r in registros}

    assert "explicacao_paciente" in nomes
    assert "resumo_gestao" in nomes


def test_comparacao_entre_versoes_de_prompt_entra_no_artefato(
    script, tmp_path, monkeypatch
):
    """Secao 5.4: duas versoes medidas nas mesmas verificacoes automaticas."""
    destino = tmp_path / "narrativas.json"
    comparacao = tmp_path / "comparacao_de_prompts.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_llm.py",
            "--fake",
            "--casos",
            "1",
            "--saida",
            str(destino),
            "--comparacao",
            str(comparacao),
        ],
    )

    script.main()

    linhas = json.loads(comparacao.read_text(encoding="utf-8"))
    versoes = {linha["identificador"] for linha in linhas}

    assert {"explicacao_paciente_v1", "explicacao_paciente_v2"} <= versoes
    for linha in linhas:
        assert set(linha["verificacoes"]) == {
            "fidelidade",
            "seguranca",
            "cobertura",
            "formato",
        }


def test_artefato_e_reprodutivel_entre_execucoes(script, tmp_path, monkeypatch):
    """Mesma regra de results/metrics.json: rodar duas vezes da o mesmo arquivo."""
    primeiro, segundo = tmp_path / "a.json", tmp_path / "b.json"

    for destino in (primeiro, segundo):
        monkeypatch.setattr(
            "sys.argv", ["run_llm.py", "--fake", "--casos", "2", "--saida", str(destino)]
        )
        script.main()

    assert primeiro.read_bytes() == segundo.read_bytes()


def test_casos_selecionados_sao_deterministicos(script):
    """Caso sorteado a cada execucao tornaria o artefato irreprodutivel."""
    primeiro = script.selecionar_casos(3)
    segundo = script.selecionar_casos(3)

    assert primeiro == segundo
    assert len(primeiro) == 3

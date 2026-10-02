"""Testes da avaliacao final dos bracos contra o conjunto de teste."""

import importlib.util
import json

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "evaluate_arms.py"
    spec = importlib.util.spec_from_file_location("evaluate_arms", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_braco_original_reproduz_o_modelo_da_fase_1(script):
    """A comparacao so e legitima se a referencia for o modelo original mesmo.

    O braco "Original" e construido a partir de `src.models.get_models()`, e nao
    de uma reconstrucao equivalente. Qualquer divergencia aqui tornaria o ganho
    reportado um artefato de implementacao.
    """
    from src import models

    bracos = script.bracos_de_referencia()

    assert "Original" in bracos
    pipeline = bracos["Original"]
    assert list(pipeline.named_steps) == ["preprocessor", "estimator"]
    original = models.get_models()[script.FAMILIA_DE_REFERENCIA]
    assert type(pipeline.named_steps["estimator"]) is type(original.named_steps["estimator"])


def test_cada_braco_de_referencia_recebe_um_pipeline_proprio(script):
    """`Pipeline` guarda estado depois do `fit`: objeto compartilhado contamina.

    Se os dois bracos apontassem para o mesmo objeto, ajustar um mudaria o outro
    sem levantar excecao, e a tabela final traria um numero estranho sem causa
    visivel.
    """
    bracos = script.bracos_de_referencia()

    assert bracos["Original"] is not bracos["Busca em grade"]


def test_limiar_dos_bracos_de_referencia_e_meio(script):
    """Os bracos que nao otimizam limiar usam 0,5, nao o 0,30 da Fase 1.

    O 0,30 da Fase 1 foi escolhido olhando o conjunto de teste, e a Fase 2 existe
    em parte para corrigir isso. Herdar aquele numero aqui contaminaria a
    comparacao que a correcao deveria tornar honesta.
    """
    assert script.LIMIAR_DE_REFERENCIA == 0.5


def test_avaliacao_reporta_custo_clinico_e_nao_so_as_metricas_tradicionais(script):
    """O enunciado pede comparacao; o projeto pede que ela inclua o custo."""
    resultado = script.avaliar_braco_de_referencia("Original")

    assert set(resultado) >= {
        "braco",
        "recall",
        "f1",
        "falsos_negativos",
        "falsos_positivos",
        "custo_clinico",
        "n_features",
        "limiar",
    }
    assert resultado["n_features"] == 30
    assert resultado["limiar"] == 0.5


def test_execucao_completa_grava_a_tabela_comparativa(script, tmp_path, monkeypatch):
    destino = tmp_path / "comparacao_final.json"
    monkeypatch.setattr(
        "sys.argv", ["evaluate_arms.py", "--saida", str(destino), "--somente-referencia"]
    )

    script.main()

    tabela = json.loads(destino.read_text(encoding="utf-8"))
    nomes = {linha["braco"] for linha in tabela}
    assert nomes == {"Original", "Busca em grade"}
    custos = [linha["custo_clinico"] for linha in tabela]
    assert custos == sorted(custos)

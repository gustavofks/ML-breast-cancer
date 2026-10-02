"""Testes da codificacao, sorteio e reparo do individuo."""

from src import config


def test_config_publica_os_parametros_de_custo_clinico():
    """Razao de custo e parcimonia sao politica clinica, nao valor magico no codigo."""
    assert config.CUSTO_FALSO_NEGATIVO > config.CUSTO_FALSO_POSITIVO
    assert config.CUSTO_FALSO_NEGATIVO / config.CUSTO_FALSO_POSITIVO == 10
    assert 0.0 <= config.PENALIDADE_PARCIMONIA < 1.0
    assert 0.0 < config.LIMIAR_MIN < config.LIMIAR_MAX < 1.0
    assert config.GENETIC_DIR.name == "genetic"
    assert config.GENETIC_DIR.parent == config.RESULTS_DIR
    assert len(config.GENETIC_SEEDS) == 3

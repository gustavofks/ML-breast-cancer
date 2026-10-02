"""Testes da codificacao, sorteio e reparo do individuo."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src import config
from src.data import load_dataset
from src.genetic import chromosome as cromo


@pytest.fixture(scope="module")
def colunas():
    X, _ = load_dataset()
    return list(X.columns)


def test_config_publica_os_parametros_de_custo_clinico():
    """Razao de custo e parcimonia sao politica clinica, nao valor magico no codigo."""
    assert config.CUSTO_FALSO_NEGATIVO > config.CUSTO_FALSO_POSITIVO
    assert config.CUSTO_FALSO_NEGATIVO / config.CUSTO_FALSO_POSITIVO == 10
    assert 0.0 <= config.PENALIDADE_PARCIMONIA < 1.0
    assert 0.0 < config.LIMIAR_MIN < config.LIMIAR_MAX < 1.0
    assert config.GENETIC_DIR.name == "genetic"
    assert config.GENETIC_DIR.parent == config.RESULTS_DIR
    assert len(config.GENETIC_SEEDS) == 3


def test_espacos_cobrem_as_tres_familias_da_fase_1():
    assert set(cromo.ESPACOS) == {"Regressão Logística", "KNN", "Random Forest"}
    assert set(cromo.PADROES) == set(cromo.ESPACOS)
    assert set(cromo.CONSTRUTORES) == set(cromo.ESPACOS)


@pytest.mark.parametrize("familia", ["Regressão Logística", "KNN", "Random Forest"])
def test_sorteio_devolve_individuo_dentro_do_espaco(familia):
    rng = np.random.default_rng(0)

    individuo = cromo.sortear(familia, rng)

    assert len(individuo.mascara) == 30
    assert set(individuo.hiperparametros) == {gene.nome for gene in cromo.ESPACOS[familia]}
    assert cromo.LIMIAR_MIN <= individuo.limiar <= cromo.LIMIAR_MAX


def test_sorteio_congela_segmentos_nao_otimizados():
    """Braco de ablacao: o segmento fora da busca fica no valor padrao da Fase 1."""
    rng = np.random.default_rng(0)

    individuo = cromo.sortear("KNN", rng, segmentos=frozenset({"mascara"}))

    assert individuo.hiperparametros == cromo.PADROES["KNN"]
    assert individuo.limiar == cromo.LIMIAR_PADRAO


def test_reparo_ativa_ao_menos_uma_feature():
    """Mascara zerada nao tem matriz de entrada: nenhum individuo assim chega ao fitness."""
    rng = np.random.default_rng(0)
    zerado = cromo.sortear("KNN", rng).substituir(mascara=(0,) * 30)

    reparado = cromo.reparar(zerado, rng)

    assert sum(reparado.mascara) >= 1


def test_reparo_traz_genes_de_volta_para_dentro_dos_limites():
    rng = np.random.default_rng(0)
    invalido = cromo.sortear("KNN", rng).substituir(
        limiar=3.5,
        hiperparametros={"n_neighbors": 99, "weights": "inexistente", "p": 2},
    )

    reparado = cromo.reparar(invalido, rng)

    assert reparado.limiar == cromo.LIMIAR_MAX
    assert reparado.hiperparametros["n_neighbors"] == 31
    assert reparado.hiperparametros["weights"] in ("uniform", "distance")


def test_gene_inteiro_com_passo_dois_so_produz_valores_impares():
    """`n_neighbors` par empata a votacao do KNN em problema binario."""
    rng = np.random.default_rng(7)

    for _ in range(20):
        individuo = cromo.reparar(cromo.sortear("KNN", rng), rng)
        assert individuo.hiperparametros["n_neighbors"] % 2 == 1


def test_decodificacao_devolve_pipeline_da_fase_1_e_features_selecionadas(colunas):
    rng = np.random.default_rng(1)
    individuo = cromo.reparar(cromo.sortear("Regressão Logística", rng), rng)

    pipeline, selecionadas = cromo.decodificar(individuo, colunas)

    assert isinstance(pipeline, Pipeline)
    # O pre-processamento vem de build_pipeline: o AG nao recria treino nenhum.
    assert list(pipeline.named_steps) == ["preprocessor", "estimator"]
    assert selecionadas == [c for c, ativo in zip(colunas, individuo.mascara) if ativo]


def test_individuo_serializa_para_json_legivel(colunas):
    rng = np.random.default_rng(2)
    individuo = cromo.reparar(cromo.sortear("Random Forest", rng), rng)

    payload = cromo.para_dicionario(individuo, colunas)

    assert payload["familia"] == "Random Forest"
    assert payload["n_features"] == sum(individuo.mascara)
    assert len(payload["features"]) == payload["n_features"]
    assert payload["limiar"] == round(individuo.limiar, 4)


def test_chave_distingue_individuos_diferentes_e_iguala_identicos():
    """A chave e o que permite memorizar o fitness sem reavaliar o mesmo individuo."""
    rng = np.random.default_rng(3)
    individuo = cromo.reparar(cromo.sortear("KNN", rng), rng)
    gemeo = individuo.substituir()
    outro = individuo.substituir(limiar=individuo.limiar + 0.1)

    assert cromo.chave(individuo) == cromo.chave(gemeo)
    assert cromo.chave(individuo) != cromo.chave(outro)

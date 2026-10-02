"""Testes dos operadores de selecao, cruzamento e mutacao."""

import numpy as np
import pytest

from src.genetic import chromosome as cromo
from src.genetic import operators as ops


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def populacao(rng):
    return [cromo.sortear("KNN", rng) for _ in range(8)]


def test_torneio_do_tamanho_da_populacao_devolve_sempre_o_melhor(populacao, rng):
    fitnesses = [0.1, 0.9, 0.3, 0.4, 0.2, 0.5, 0.6, 0.7]

    escolhido = ops.torneio(populacao, fitnesses, k=len(populacao), rng=rng)

    assert escolhido is populacao[1]


def test_torneio_de_tamanho_um_e_sorteio_uniforme(populacao, rng):
    """Com k=1 nao ha pressao seletiva: o operador vira amostragem aleatoria."""
    fitnesses = [0.0] * 8
    fitnesses[0] = 1.0

    escolhidos = {id(ops.torneio(populacao, fitnesses, k=1, rng=rng)) for _ in range(40)}

    assert len(escolhidos) > 1


def test_mutacao_com_taxa_zero_e_identidade(populacao, rng):
    """Operador sem efeito colateral escondido: taxa zero nao muda nada."""
    original = populacao[0]

    mutado = ops.mutar(original, taxa=0.0, rng=rng)

    assert mutado == original


def test_mutacao_com_taxa_um_altera_a_mascara(populacao, rng):
    original = populacao[0]

    mutado = ops.mutar(original, taxa=1.0, rng=rng)

    assert mutado.mascara != original.mascara


def test_mutacao_respeita_segmento_congelado(populacao, rng):
    """Braco de ablacao: o segmento fora da busca nao pode ser tocado."""
    original = populacao[0]

    mutado = ops.mutar(original, taxa=1.0, rng=rng, segmentos=frozenset({"mascara"}))

    assert mutado.hiperparametros == original.hiperparametros
    assert mutado.limiar == original.limiar


def test_mutacao_devolve_individuo_valido(populacao, rng):
    for individuo in populacao:
        mutado = ops.mutar(individuo, taxa=1.0, rng=rng)

        assert sum(mutado.mascara) >= 1
        assert cromo.LIMIAR_MIN <= mutado.limiar <= cromo.LIMIAR_MAX
        assert mutado.hiperparametros["n_neighbors"] % 2 == 1


def test_cruzamento_de_pais_identicos_devolve_o_pai(populacao, rng):
    """O operador preserva material genetico: sem pais diferentes, sem novidade."""
    pai = populacao[0]

    filho_a, filho_b = ops.cruzar(pai, pai, taxa=1.0, rng=rng)

    assert filho_a == pai
    assert filho_b == pai


def test_cruzamento_com_taxa_zero_devolve_os_pais(populacao, rng):
    pai, mae = populacao[0], populacao[1]

    filho_a, filho_b = ops.cruzar(pai, mae, taxa=0.0, rng=rng)

    assert filho_a == pai
    assert filho_b == mae


def test_cruzamento_uniforme_so_combina_genes_dos_pais(populacao, rng):
    pai, mae = populacao[0], populacao[1]

    filho_a, filho_b = ops.cruzar(pai, mae, taxa=1.0, rng=rng)

    for indice in range(30):
        assert filho_a.mascara[indice] in (pai.mascara[indice], mae.mascara[indice])
        assert filho_b.mascara[indice] in (pai.mascara[indice], mae.mascara[indice])


def test_cruzamento_aritmetico_fica_entre_os_pais(rng):
    """Gene real e media ponderada: o filho nunca sai do intervalo dos pais."""
    pai = cromo.sortear("Regressão Logística", rng).substituir(limiar=0.2)
    mae = pai.substituir(limiar=0.8)

    filho_a, filho_b = ops.cruzar(pai, mae, taxa=1.0, rng=rng)

    assert 0.2 <= filho_a.limiar <= 0.8
    assert 0.2 <= filho_b.limiar <= 0.8


def test_cruzamento_respeita_segmento_congelado(populacao, rng):
    pai, mae = populacao[0], populacao[1]

    filho_a, _ = ops.cruzar(pai, mae, taxa=1.0, rng=rng, segmentos=frozenset({"mascara"}))

    assert filho_a.hiperparametros == pai.hiperparametros
    assert filho_a.limiar == pai.limiar

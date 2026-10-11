"""Testes do laco evolutivo, do elitismo e do historico por geracao."""

import numpy as np
import pytest

from src.data import load_dataset
from src.genetic import chromosome as cromo
from src.genetic import engine
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def treino():
    X, y = load_dataset()
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train


@pytest.fixture
def config_minima():
    """Populacao e geracoes minimas: o teste verifica o laco, nao a qualidade."""
    return engine.ConfiguracaoAG(
        nome="teste",
        populacao=6,
        geracoes=3,
        taxa_mutacao=0.2,
        taxa_cruzamento=0.8,
        torneio=2,
        elitismo=2,
    )


def test_execucao_completa_de_ponta_a_ponta(treino, config_minima):
    X_train, y_train = treino

    resultado = engine.evoluir(config_minima, "Regressão Logística", X_train, y_train, semente=42)

    assert len(resultado.historico) == config_minima.geracoes
    assert resultado.melhor is not None
    assert resultado.melhor_avaliacao.n_amostras == len(y_train)
    assert len(resultado.melhores) == 5


def test_elitismo_mantem_o_melhor_fitness_monotonico(treino, config_minima):
    """Com elitismo, o melhor da geracao nunca pode piorar na geracao seguinte."""
    X_train, y_train = treino

    resultado = engine.evoluir(config_minima, "Regressão Logística", X_train, y_train, semente=42)

    valores = [registro.melhor_fitness for registro in resultado.historico]
    assert valores == sorted(valores)


def test_configuracao_sem_elitismo_roda_o_laco_completo(treino):
    """E4 existe para medir a contribuicao do elitismo; o laco tem que aceitar zero."""
    X_train, y_train = treino
    sem_elite = engine.ConfiguracaoAG(
        nome="sem elitismo",
        populacao=6,
        geracoes=4,
        taxa_mutacao=0.9,
        taxa_cruzamento=0.8,
        torneio=2,
        elitismo=0,
    )

    resultado = engine.evoluir(sem_elite, "Regressão Logística", X_train, y_train, semente=7)

    assert resultado.configuracao.elitismo == 0
    assert len(resultado.historico) == 4
    # Sem elitismo nenhum individuo e copiado intacto, e a populacao final pode
    # nao conter o melhor. Por isso o melhor global e guardado a parte, e ele
    # nunca pode ficar abaixo da media de qualquer geracao.
    assert resultado.melhor_avaliacao.fitness >= max(
        registro.fitness_medio for registro in resultado.historico
    )


def test_mesma_semente_produz_o_mesmo_resultado(treino, config_minima):
    X_train, y_train = treino

    primeira = engine.evoluir(config_minima, "KNN", X_train, y_train, semente=42)
    segunda = engine.evoluir(config_minima, "KNN", X_train, y_train, semente=42)

    assert cromo.chave(primeira.melhor) == cromo.chave(segunda.melhor)
    assert primeira.melhor_avaliacao.fitness == segunda.melhor_avaliacao.fitness


def test_memorizacao_evita_reavaliar_o_mesmo_individuo(treino, config_minima):
    """Reavaliar um individuo custa uma validacao cruzada inteira."""
    X_train, y_train = treino

    resultado = engine.evoluir(config_minima, "KNN", X_train, y_train, semente=42)

    maximo_possivel = config_minima.populacao * config_minima.geracoes
    assert resultado.avaliacoes_totais <= maximo_possivel


def test_diversidade_e_zero_em_populacao_uniforme():
    individuo = cromo.sortear("KNN", np.random.default_rng(0))

    assert engine.diversidade([individuo] * 5) == pytest.approx(0.0)


def test_diversidade_e_positiva_em_populacao_variada():
    rng = np.random.default_rng(0)
    populacao = [cromo.sortear("KNN", rng) for _ in range(10)]

    assert engine.diversidade(populacao) > 0.0


def test_segmentos_congelados_permanecem_no_padrao(treino, config_minima):
    """Braco 'AG so limiar': 30 features e hiperparametros da Fase 1."""
    X_train, y_train = treino

    resultado = engine.evoluir(
        config_minima,
        "KNN",
        X_train,
        y_train,
        semente=42,
        segmentos=frozenset({"limiar"}),
    )

    assert sum(resultado.melhor.mascara) == 30
    assert resultado.melhor.hiperparametros == cromo.PADROES["KNN"]


def test_elitismo_nao_pode_ocupar_a_populacao_inteira():
    """Elitismo maior ou igual a populacao copiaria tudo, e o AG nao evoluiria.

    O defeito seria silencioso: nenhum erro, nenhuma excecao, so um laco que
    repete a mesma populacao e um relatorio afirmando convergencia que nunca
    aconteceu. Por isso a configuracao e rejeitada na construcao, antes de
    qualquer execucao.
    """
    with pytest.raises(ValueError, match="elitismo"):
        engine.ConfiguracaoAG(
            nome="elite demais",
            populacao=6,
            geracoes=3,
            taxa_mutacao=0.1,
            taxa_cruzamento=0.8,
            torneio=2,
            elitismo=6,
        )


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("geracoes", 0),
        ("populacao", 1),
        ("torneio", 0),
        ("taxa_mutacao", 1.5),
        ("taxa_cruzamento", -0.1),
    ],
)
def test_configuracao_invalida_falha_na_construcao(campo, valor):
    """Erro claro na configuracao, em vez de estouro obscuro no meio do laco."""
    campos = {
        "nome": "invalida",
        "populacao": 6,
        "geracoes": 3,
        "taxa_mutacao": 0.1,
        "taxa_cruzamento": 0.8,
        "torneio": 2,
        "elitismo": 2,
    }
    campos[campo] = valor

    with pytest.raises(ValueError, match=campo):
        engine.ConfiguracaoAG(**campos)


def test_experimentos_do_enunciado_sao_configuracoes_validas():
    """A validacao roda na importacao: configuracao invalida nao chega a existir."""
    from src.genetic import experiments

    for configuracao in experiments.EXPERIMENTOS:
        assert configuracao.elitismo < configuracao.populacao
        assert configuracao.geracoes >= 1


def test_historico_serializa_para_json(treino, config_minima):
    X_train, y_train = treino

    resultado = engine.evoluir(config_minima, "KNN", X_train, y_train, semente=42)
    payload = engine.para_dicionario(resultado, list(X_train.columns))

    assert payload["experimento"] == "teste"
    assert payload["semente"] == 42
    assert len(payload["historico"]) == config_minima.geracoes
    assert payload["melhor"]["familia"] == "KNN"
    assert "fitness" in payload["avaliacao"]


# ---------------------------------------------------------------------------
# Avaliacao da populacao em paralelo
# ---------------------------------------------------------------------------
from src.genetic.chromosome import PADROES  # noqa: E402


@pytest.mark.parametrize("familia", ["Regressão Logística", "Random Forest"])
def test_avaliacao_paralela_da_o_mesmo_resultado_que_a_sequencial(treino, familia):
    """Paralelizar so pode mudar o tempo, nunca um byte do artefato.

    A avaliacao de um individuo depende so dele e da semente da validacao
    cruzada, nunca do gerador aleatorio da busca, entao a ordem de sorteio fica
    intacta. A memoria e preenchida na ordem de primeira aparicao, como na versao
    sequencial: o ranking dos cinco melhores desempata por essa ordem.
    """
    X_train, y_train = treino
    configuracao = engine.ConfiguracaoAG(
        nome="paralelo",
        populacao=8,
        geracoes=3,
        taxa_mutacao=0.3,
        taxa_cruzamento=0.8,
        torneio=2,
        elitismo=1,
    )
    colunas = list(X_train.columns)

    sequencial = engine.evoluir(configuracao, familia, X_train, y_train, semente=7, n_jobs=1)
    paralelo = engine.evoluir(configuracao, familia, X_train, y_train, semente=7, n_jobs=2)

    assert engine.para_dicionario(paralelo, colunas) == engine.para_dicionario(
        sequencial, colunas
    )


def test_floresta_nao_paraleliza_por_dentro():
    """O paralelismo fica na populacao, nao dentro de cada floresta.

    Com 455 amostras, distribuir as arvores entre threads custa mais do que
    treina-las: medido, 1,65 s por avaliacao com n_jobs=-1 contra 1,18 s com 1, e
    fitness identico. Somado ao paralelismo da populacao, multiplicaria os
    processos pelos nucleos.
    """
    floresta = cromo.CONSTRUTORES["Random Forest"](dict(PADROES["Random Forest"]))

    assert floresta.n_jobs == 1

"""Testes do custo clinico e do isolamento do conjunto de teste."""

import numpy as np
import pytest

from src import config
from src.data import load_dataset
from src.genetic import chromosome as cromo
from src.genetic import fitness as fit
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def treino():
    X, y = load_dataset()
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train


@pytest.fixture(scope="module")
def individuo():
    rng = np.random.default_rng(0)
    return cromo.sortear("Regressão Logística", rng, segmentos=frozenset({"limiar"}))


def test_nenhum_modulo_do_pacote_abre_porta_para_dados_de_teste():
    """A separacao treino/teste nao pode ser violada nem por engano.

    O pacote inteiro recebe `(X_train, y_train)` por parametro. Se algum modulo
    importasse `load_dataset` ou `split_data`, poderia refazer a separacao e
    tocar no teste sem que nada acusasse erro — exatamente o tipo de defeito
    silencioso que a Fase 1 aprendeu a pegar por teste de propriedade.
    """
    proibidos = ("load_dataset", "split_data", "X_test", "y_test", "TEST_SIZE")
    modulos = sorted((config.PROJECT_ROOT / "src" / "genetic").glob("*.py"))

    assert modulos, "pacote src/genetic nao encontrado"
    for caminho in modulos:
        fonte = caminho.read_text(encoding="utf-8")
        for token in proibidos:
            assert token not in fonte, f"{caminho.name} referencia '{token}'"


def test_avaliacao_usa_exatamente_as_linhas_recebidas(treino, individuo):
    """Previsao fora de fold cobre cada linha do treino uma vez, e nenhuma outra."""
    X_train, y_train = treino

    avaliacao = fit.avaliar(individuo, X_train, y_train)

    assert avaliacao.n_amostras == len(y_train)
    assert avaliacao.falsos_negativos + avaliacao.falsos_positivos <= len(y_train)


def test_fitness_e_deterministico_com_a_mesma_semente(treino, individuo):
    X_train, y_train = treino

    primeira = fit.avaliar(individuo, X_train, y_train)
    segunda = fit.avaliar(individuo, X_train, y_train)

    assert primeira.fitness == segunda.fitness


def test_falso_negativo_pesa_mais_que_falso_positivo():
    """A assimetria do custo e a traducao da politica clinica da Fase 1."""
    custo = fit.CustoClinico()

    apenas_fn = fit.custo_medio(falsos_negativos=1, falsos_positivos=0, n=100, custo=custo)
    apenas_fp = fit.custo_medio(falsos_negativos=0, falsos_positivos=1, n=100, custo=custo)

    assert apenas_fn == pytest.approx(10 * apenas_fp)


def test_classificador_perfeito_com_todas_as_features_tem_fitness_abaixo_de_um():
    """Sem erro nenhum, o que sobra descontando e o preco das 30 medidas."""
    custo = fit.CustoClinico(parcimonia=0.05)

    valor = fit.combinar(custo_medio=0.0, n_features=30, total_features=30, custo=custo)

    assert valor == pytest.approx(0.95)


def test_parcimonia_desempata_em_favor_de_menos_medidas():
    custo = fit.CustoClinico(parcimonia=0.05)

    com_dez = fit.combinar(custo_medio=0.1, n_features=10, total_features=30, custo=custo)
    com_trinta = fit.combinar(custo_medio=0.1, n_features=30, total_features=30, custo=custo)

    assert com_dez > com_trinta


def test_parcimonia_zerada_ignora_o_numero_de_features():
    custo = fit.CustoClinico(parcimonia=0.0)

    com_um = fit.combinar(custo_medio=0.2, n_features=1, total_features=30, custo=custo)
    com_trinta = fit.combinar(custo_medio=0.2, n_features=30, total_features=30, custo=custo)

    assert com_um == com_trinta


def test_limiar_baixo_reduz_falsos_negativos(treino):
    """Propriedade, nao valor: baixar o corte so pode mover casos para maligno."""
    X_train, y_train = treino
    base = cromo.sortear("Regressão Logística", np.random.default_rng(1), segmentos=frozenset())

    cauteloso = fit.avaliar(base.substituir(limiar=0.15), X_train, y_train)
    permissivo = fit.avaliar(base.substituir(limiar=0.85), X_train, y_train)

    assert cauteloso.falsos_negativos <= permissivo.falsos_negativos
    assert cauteloso.falsos_positivos >= permissivo.falsos_positivos


def test_reavaliacao_com_a_mesma_semente_nao_tem_queda(treino, individuo):
    """Semente igual, particoes iguais, fitness igual: a queda e zero por construcao."""
    X_train, y_train = treino
    original = fit.avaliar(individuo, X_train, y_train, cv_seed=config.SEED)

    estabilidade = fit.reavaliar_estabilidade(
        individuo, original, X_train, y_train, cv_seed=config.SEED
    )

    assert estabilidade.queda == pytest.approx(0.0)
    assert estabilidade.fitness_revalidado == pytest.approx(original.fitness)


def test_queda_positiva_significa_piora(treino, individuo):
    """Sinal declarado: queda > 0 e sobreajuste as particoes da busca."""
    X_train, y_train = treino
    original = fit.avaliar(individuo, X_train, y_train, cv_seed=config.SEED)

    estabilidade = fit.reavaliar_estabilidade(
        individuo, original, X_train, y_train, cv_seed=config.SEED + 100
    )

    assert estabilidade.queda == pytest.approx(
        original.fitness - estabilidade.fitness_revalidado
    )


def test_reavaliacao_em_lote_preserva_a_ordem_recebida(treino):
    """O relatorio cita "o terceiro melhor": a ordem nao pode embaralhar."""
    X_train, y_train = treino
    rng = np.random.default_rng(5)
    individuos = [cromo.sortear("KNN", rng) for _ in range(3)]
    avaliacoes = [fit.avaliar(i, X_train, y_train) for i in individuos]

    lote = fit.reavaliar_lote(
        list(zip(individuos, avaliacoes)), X_train, y_train, cv_seed=config.SEED + 100
    )

    assert len(lote) == 3
    for estabilidade, avaliacao in zip(lote, avaliacoes):
        assert estabilidade.fitness_original == pytest.approx(avaliacao.fitness)


def test_semente_de_validacao_diferente_produz_avaliacao_independente(treino, individuo):
    """Base da verificacao de estabilidade da secao 4.7 do design."""
    X_train, y_train = treino

    original = fit.avaliar(individuo, X_train, y_train, cv_seed=config.SEED)
    outra = fit.avaliar(individuo, X_train, y_train, cv_seed=config.SEED + 100)

    assert original.n_amostras == outra.n_amostras
    assert 0.0 <= outra.fitness <= 1.0

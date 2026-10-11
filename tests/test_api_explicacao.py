"""Testes das contribuicoes por caso."""

import numpy as np
import pytest

from src import explain
from src.api import explicacao, servico
from src.data import load_dataset
from src.genetic.chromosome import PADROES
from src.preprocessing import split_data

TRES_MEDIDAS = ["texture_worst", "area_worst", "radius_mean"]


def _servico(familia, features=None):
    dados = servico.referencia_da_fase_1().para_dicionario()
    dados.update(familia=familia, hiperparametros=dict(PADROES[familia]))
    if features:
        dados["features"] = features
    return servico.Servico.construir(servico.EspecificacaoDoModelo.de_dicionario(dados))


@pytest.fixture(scope="module")
def caso():
    X, y = load_dataset()
    _, X_test, _, _ = split_data(X, y)
    return X_test.iloc[[0]]


@pytest.mark.parametrize("familia", ["Regressão Logística", "KNN", "Random Forest"])
def test_uma_contribuicao_por_medida_em_toda_familia(familia, caso):
    """Random Forest devolve uma fatia por classe; KNN nao tem explicador direto.

    Nenhum dos dois pode virar lista com o tamanho errado: o emparelhamento com
    os nomes truncaria em silencio, e o texto narraria a medida errada.
    """
    modelo = _servico(familia, features=TRES_MEDIDAS)

    pares = explicacao.contribuicoes(modelo, caso)

    assert [nome for nome, _ in pares] == modelo.selecionadas
    assert all(np.isfinite(valor) for _, valor in pares)


def test_floresta_usa_a_fatia_da_classe_maligna(caso):
    """Pegar a classe 0 inverteria o sinal de toda a explicacao."""
    modelo = _servico("Random Forest", features=TRES_MEDIDAS)
    pares = dict(explicacao.contribuicoes(modelo, caso))

    bruto = explain.shap_explanation(
        modelo.pipeline, modelo.X_train[modelo.selecionadas], caso[modelo.selecionadas]
    )
    esperado = np.asarray(bruto.values)[0, :, 1]

    assert [pares[n] for n in modelo.selecionadas] == pytest.approx(list(esperado))


def test_regressao_logistica_coincide_com_o_shap_da_fase_1(caso):
    """O texto narra a mesma explicacao que o grafico da Fase 1 mostra."""
    modelo = _servico("Regressão Logística")
    pares = dict(explicacao.contribuicoes(modelo, caso))

    bruto = explain.shap_explanation(modelo.pipeline, modelo.X_train, caso)

    assert [pares[n] for n in modelo.selecionadas] == pytest.approx(
        list(np.asarray(bruto.values)[0])
    )


def test_contribuicoes_da_permutacao_sao_deterministicas(caso):
    """O explicador por permutacao e estocastico; com semente fixa, nao pode variar."""
    modelo = _servico("KNN", features=TRES_MEDIDAS)

    assert explicacao.contribuicoes(modelo, caso) == explicacao.contribuicoes(modelo, caso)


def test_caso_com_medidas_a_mais_e_recortado(caso):
    """O caso chega com as 30 medidas; o modelo de tres explica so as suas."""
    modelo = _servico("Regressão Logística", features=TRES_MEDIDAS)

    assert len(explicacao.contribuicoes(modelo, caso)) == 3

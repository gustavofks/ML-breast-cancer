"""Testes do nucleo de dominio da API: o que e servido e como se preve."""

import json

import pytest

from src import config
from src.api import servico


def test_sem_arquivo_serve_a_referencia_da_fase_1_e_diz_isso(tmp_path):
    """Servir a referencia em silencio faria /model-info mentir sobre a origem."""
    especificacao = servico.carregar_especificacao(tmp_path / "nao_existe.json")

    assert especificacao == servico.referencia_da_fase_1()
    assert "Fase 1" in especificacao.origem


def test_referencia_usa_as_30_medidas_e_o_limiar_padrao():
    referencia = servico.referencia_da_fase_1()

    assert referencia.familia == "Regressão Logística"
    assert len(referencia.features) == config.EXPECTED_N_FEATURES
    assert referencia.limiar == 0.5


def test_arquivo_no_formato_do_melhor_individuo_e_aceito(tmp_path):
    """O usuario copia o `melhor` de uma execucao do AG, sem traducao."""
    caminho = tmp_path / "modelo.json"
    caminho.write_text(
        json.dumps(
            {
                "familia": "Regressão Logística",
                "hiperparametros": {"C": 0.5, "class_weight": "balanced"},
                "features": ["texture_worst", "area_worst"],
                "limiar": 0.3,
                "origem": "AG, completo, E1, semente 42",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    especificacao = servico.carregar_especificacao(caminho)

    assert especificacao.features == ("texture_worst", "area_worst")
    assert especificacao.limiar == 0.3


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("familia", "SVM"),
        ("features", []),
        ("features", ["nao_existe"]),
        ("limiar", 1.5),
        ("limiar", 0.0),
    ],
)
def test_especificacao_invalida_falha_na_carga(campo, valor):
    """Erro na inicializacao, nao na primeira requisicao de um medico."""
    dados = servico.referencia_da_fase_1().para_dicionario()
    dados[campo] = valor

    with pytest.raises(ValueError):
        servico.EspecificacaoDoModelo.de_dicionario(dados)


def test_especificacao_ida_e_volta_pelo_dicionario_e_identidade():
    referencia = servico.referencia_da_fase_1()

    assert servico.EspecificacaoDoModelo.de_dicionario(referencia.para_dicionario()) == referencia

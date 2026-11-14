"""Testes dos casos PAAF da clinica ficticia."""

from src import config
from src.clinica import paaf


def test_casos_sao_os_114_do_teste_da_fase2_na_mesma_ordem_da_vitrine():
    casos = paaf.casos_de_teste()

    assert len(casos) == 114
    assert [c.caso for c in casos] == list(range(114))


def test_cada_caso_traz_as_30_medidas_e_uma_referencia_valida():
    for caso in paaf.casos_de_teste():
        assert len(caso.medidas) == config.EXPECTED_N_FEATURES
        assert caso.referencia in config.CLASS_NAMES


def test_proporcao_de_malignos_preserva_a_estratificacao():
    referencias = [c.referencia for c in paaf.casos_de_teste()]

    assert referencias.count("Maligno") == 42

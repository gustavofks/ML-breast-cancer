"""Formato dos numeros mostrados ao publico."""

from __future__ import annotations

from src.ui.formato import num, pct, por_nome


def test_numero_usa_virgula_decimal():
    assert num(0.9286, 3) == "0,929"
    assert num(10.0, 0) == "10"


def test_percentual_multiplica_e_arredonda():
    assert pct(0.9286) == "92,9%"
    assert pct(0.9, 0) == "90%"


def test_busca_por_nome_devolve_a_linha_certa():
    linhas = [{"modelo": "A", "f1": 1}, {"modelo": "B", "f1": 2}]
    assert por_nome(linhas, "modelo", "B")["f1"] == 2

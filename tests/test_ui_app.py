"""Testes do painel: logica pura primeiro, a tela do Gradio por ultimo."""

import matplotlib

matplotlib.use("Agg")  # backend sem janela: o teste roda em terminal

import pytest  # noqa: E402

from src import config  # noqa: E402
from src.plotting import COLOR_BENIGNO, COLOR_MALIGNO  # noqa: E402
from src.ui import app as painel  # noqa: E402


def test_casos_de_teste_sao_o_conjunto_de_teste_com_o_rotulo_registrado():
    """O painel mostra acerto e erro; para isso precisa do diagnostico registrado."""
    casos = painel.casos_de_teste()

    assert len(casos) == 114
    assert set(casos["diagnostico_registrado"]) == set(config.CLASS_NAMES)
    assert len(casos.columns) == config.EXPECTED_N_FEATURES + 1


def test_rotulo_do_caso_nomeia_indice_e_diagnostico_registrado():
    casos = painel.casos_de_teste()

    rotulo = painel.rotulo_do_caso(casos, 0)

    assert rotulo.startswith("Caso 0")
    assert casos["diagnostico_registrado"].iloc[0] in rotulo


CONTRIBUICOES = [
    {"feature": f"medida_{i}", "contribuicao": valor}
    for i, valor in enumerate([2.0, -1.5, 1.0, -0.8, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1, 0.05, -0.01])
]


def test_grafico_mostra_as_dez_maiores_contribuicoes():
    figura = painel.figura_de_contribuicoes(CONTRIBUICOES)
    barras = figura.axes[0].patches

    assert len(barras) == 10


def test_maior_contribuicao_fica_no_topo_do_grafico():
    """barh desenha de baixo para cima; sem inverter, a mais importante fica escondida no pe."""
    figura = painel.figura_de_contribuicoes(CONTRIBUICOES)
    rotulos = [t.get_text() for t in figura.axes[0].get_yticklabels()]

    assert rotulos[-1] == "medida_0"


def test_cor_da_barra_segue_o_sinal_da_contribuicao():
    """Laranja empurra para maligno, azul para benigno: as cores da Fase 1."""
    figura = painel.figura_de_contribuicoes(CONTRIBUICOES[:2])
    cores = {
        round(b.get_width(), 6): matplotlib.colors.to_hex(b.get_facecolor())
        for b in figura.axes[0].patches
    }

    assert cores[2.0] == matplotlib.colors.to_hex(COLOR_MALIGNO)
    assert cores[-1.5] == matplotlib.colors.to_hex(COLOR_BENIGNO)

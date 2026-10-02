"""Testes das figuras do algoritmo genetico."""

import matplotlib

matplotlib.use("Agg")  # backend sem janela: o teste roda em terminal

from src.genetic import plots


def test_curva_de_convergencia_tem_uma_linha_por_execucao(payload_ag):
    payloads = [payload_ag(semente=42), payload_ag(semente=43)]

    figura = plots.figura_convergencia(payloads)
    eixo = figura.axes[0]

    assert len(eixo.lines) >= len(payloads)
    assert eixo.get_xlabel()
    assert eixo.get_ylabel()


def test_fronteira_relaciona_custo_e_numero_de_features(payload_ag):
    payloads = [
        payload_ag(features=("a",), fn=8),
        payload_ag(features=("a", "b", "c"), fn=2),
    ]

    figura = plots.figura_fronteira(payloads)
    eixo = figura.axes[0]

    assert eixo.collections
    assert "features" in eixo.get_xlabel().lower()


def test_mapa_de_frequencia_marca_as_features_do_topo_shap(payload_ag):
    shap_global = [
        {"feature": "texture_worst", "shap_medio": 1.25},
        {"feature": "area_worst", "shap_medio": 0.8},
    ]
    payloads = [payload_ag(features=("texture_worst", "area_worst"))]

    figura = plots.figura_frequencia(payloads, shap_global)
    eixo = figura.axes[0]

    rotulos = [texto.get_text() for texto in eixo.get_yticklabels()]
    assert "texture_worst" in rotulos


def test_figuras_sao_gravadas_no_diretorio_do_algoritmo_genetico(payload_ag, tmp_path):
    """A secao 4.8 do design mantem os artefatos do AG juntos, em results/genetic/."""
    figura = plots.figura_convergencia([payload_ag()])

    caminho = plots.salvar(figura, "teste.png", destino=tmp_path)

    assert caminho.exists()
    assert caminho.parent == tmp_path

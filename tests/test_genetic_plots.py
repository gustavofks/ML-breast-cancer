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


def test_legenda_traz_uma_entrada_por_experimento_e_nao_por_execucao(payload_ag):
    """A execucao completa tem 45 buscas: legenda com 45 entradas nao se le.

    Cada semente continua desenhada — a dispersao entre elas e informacao — mas a
    legenda identifica o experimento, que e a variavel que o leitor compara.
    """
    payloads = [
        payload_ag(experimento="E1 base", semente=semente) for semente in (42, 43, 44)
    ] + [payload_ag(experimento="E2 exploração", semente=semente) for semente in (42, 43)]

    figura = plots.figura_convergencia(payloads)
    eixo = figura.axes[0]

    rotulos = [texto.get_text() for texto in eixo.get_legend().get_texts()]

    assert len(eixo.lines) == 5
    assert sorted(rotulos) == ["E1 base", "E2 exploração"]


def test_execucoes_do_mesmo_experimento_compartilham_a_cor(payload_ag):
    """Cor e o que liga as sementes de um experimento no grafico."""
    payloads = [payload_ag(experimento="E1 base", semente=s) for s in (42, 43)]

    figura = plots.figura_convergencia(payloads)
    cores = {linha.get_color() for linha in figura.axes[0].lines}

    assert len(cores) == 1


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


def test_varredura_desenha_uma_curva_por_parcimonia_em_cada_painel(payload_ag):
    """Eixo x e a razao de custo; cada linha e um alfa. Limiar e medidas, lado a lado."""
    payloads = []
    for razao in (5, 10, 20):
        for alfa in (0.0, 0.15):
            payload = payload_ag(features=("a",) * (1 + razao // 5))
            payload["melhor"]["limiar"] = 0.5 - razao / 100
            payload["custo"] = {
                "falso_negativo": float(razao),
                "falso_positivo": 1.0,
                "parcimonia": alfa,
            }
            payloads.append(payload)

    figura = plots.figura_varredura(payloads)
    eixos = figura.axes

    assert len(eixos) == 2
    for eixo in eixos:
        assert len(eixo.get_lines()) == 2
        for linha in eixo.get_lines():
            assert list(linha.get_xdata()) == [5.0, 10.0, 20.0]

"""Testes das configuracoes exigidas pelo enunciado."""

from src import config
from src.genetic import experiments as exp
from src.genetic.chromosome import SEGMENTOS


def test_o_enunciado_pede_tres_configuracoes_e_entregamos_quatro():
    assert len(exp.EXPERIMENTOS) == 4
    assert [c.nome for c in exp.EXPERIMENTOS] == [
        "E1 base",
        "E2 exploração",
        "E3 explotação",
        "E4 sem elitismo",
    ]


def test_configuracoes_sao_de_fato_diferentes_entre_si():
    """Experimento que repete outro nao responde pergunta nenhuma."""
    assinaturas = {
        (c.populacao, c.geracoes, c.taxa_mutacao, c.taxa_cruzamento, c.torneio, c.elitismo)
        for c in exp.EXPERIMENTOS
    }

    assert len(assinaturas) == len(exp.EXPERIMENTOS)


def test_e4_e_igual_a_e1_exceto_pelo_elitismo():
    """So assim a diferenca observada pode ser atribuida ao elitismo."""
    e1, e4 = exp.EXPERIMENTOS[0], exp.EXPERIMENTOS[3]

    assert (e1.populacao, e1.geracoes, e1.taxa_mutacao, e1.taxa_cruzamento, e1.torneio) == (
        e4.populacao,
        e4.geracoes,
        e4.taxa_mutacao,
        e4.taxa_cruzamento,
        e4.torneio,
    )
    assert e1.elitismo == 2
    assert e4.elitismo == 0


def test_bracos_de_ablacao_isolam_um_segmento_cada():
    """Ganho que vem so do limiar nao pode ser creditado ao AG inteiro."""
    assert exp.BRACOS["AG, só limiar"] == frozenset({"limiar"})
    assert exp.BRACOS["AG, só features"] == frozenset({"mascara"})
    assert exp.BRACOS["AG, só hiperparâmetros"] == frozenset({"hiperparametros"})
    assert exp.BRACOS["AG, completo"] == SEGMENTOS


def test_varredura_de_sensibilidade_cobre_as_razoes_e_alfas_do_design():
    razoes = {custo.falso_negativo / custo.falso_positivo for custo in exp.VARREDURA_CUSTO}
    alfas = {round(custo.parcimonia, 4) for custo in exp.VARREDURA_CUSTO}

    assert razoes == {5.0, 10.0, 20.0}
    assert alfas == {0.0, 0.05, 0.15}
    assert len(exp.VARREDURA_CUSTO) == 9


def test_sementes_vem_da_configuracao_central():
    assert exp.SEMENTES == config.GENETIC_SEEDS

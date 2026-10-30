"""Testes da montagem do contexto clinico e da geracao dos tres textos."""

import pytest

from src.llm import narrator
from src.llm import prompts
from src.llm.provider import FakeProvider

CONTRIBUICOES = [
    ("texture_worst", 1.25),
    ("area_worst", 0.80),
    ("concave points_mean", 0.64),
    ("radius_mean", 0.10),
]

TEXTO_VALIDO = """
## Leitura do caso
A probabilidade estimada é 0,91, acima do limiar de 0,25.

## Medidas que mais pesaram
Pesaram texture_worst, area_worst e concave points_mean.

## Ressalva
Esta leitura é apoio à decisão e não substitui a avaliação do médico responsável.
"""


def test_contexto_do_paciente_carrega_so_o_que_o_texto_pode_citar():
    """O que nao esta no contexto nao pode aparecer no texto, por construcao."""
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.91, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    assert contexto["probabilidade"] == 0.91
    assert contexto["limiar"] == 0.25
    assert contexto["decisao"] == "maligno"


def test_contexto_traz_as_tres_maiores_contribuicoes_em_ordem():
    """A verificacao de cobertura exige as tres maiores; a ordem importa."""
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.91, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    nomes = [c["feature"] for c in contexto["maiores_contribuicoes"]]

    assert nomes == ["texture_worst", "area_worst", "concave points_mean"]


def test_decisao_abaixo_do_limiar_e_benigno():
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.10, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    assert contexto["decisao"] == "benigno"


def test_valores_do_contexto_varre_a_estrutura_inteira():
    """A lista de valores permitidos sai do proprio contexto, nao de uma lista a mao.

    E isso que torna a fidelidade uma propriedade e nao uma lista mantida em
    paralelo: qualquer numero que o texto cite tem de ter vindo do contexto.
    """
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.91, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    valores = narrator.valores_do_contexto(contexto)

    assert 0.91 in valores
    assert 0.25 in valores
    assert 1.25 in valores


def test_narrar_devolve_texto_identificador_versao_e_qualidade():
    """Texto sem a avaliacao ao lado convida a publicar sem medir."""
    template = prompts.TEMPLATES["explicacao_paciente_v2"]
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.91, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    narrativa = narrator.narrar(template, contexto, FakeProvider([TEXTO_VALIDO]))

    assert narrativa.texto == TEXTO_VALIDO
    assert narrativa.identificador == "explicacao_paciente_v2"
    assert narrativa.versao == 2
    assert narrativa.qualidade.aprovado


def test_narrativa_reprovada_nao_e_escondida():
    """Texto ruim entra no artefato marcado como reprovado, nao e descartado."""
    template = prompts.TEMPLATES["explicacao_paciente_v2"]
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.91, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    narrativa = narrator.narrar(
        template, contexto, FakeProvider(["O diagnóstico é câncer confirmado."])
    )

    assert not narrativa.qualidade.aprovado
    assert narrativa.texto


def test_narrativa_serializa_com_a_avaliacao_junto():
    template = prompts.TEMPLATES["explicacao_paciente_v2"]
    contexto = narrator.contexto_do_paciente(
        probabilidade=0.91, limiar=0.25, contribuicoes=CONTRIBUICOES
    )

    payload = narrator.narrar(template, contexto, FakeProvider([TEXTO_VALIDO])).para_dicionario()

    assert payload["identificador"] == "explicacao_paciente_v2"
    assert payload["qualidade"]["aprovado"] is True
    assert payload["contexto"]["probabilidade"] == 0.91


def test_contexto_de_gestao_le_o_metrics_json_da_fase_1():
    """Nenhum numero e digitado a mao, nem no caminho ate o prompt."""
    metricas = {
        "modelo_escolhido": "Regressão Logística",
        "teste": [
            {"modelo": "Regressão Logística", "recall": 0.9286, "f1": 0.9512,
             "falsos_negativos": 3, "falsos_positivos": 1},
        ],
        "dataset": {"amostras": 569},
    }

    contexto = narrator.contexto_de_gestao(metricas)

    assert contexto["modelo"] == "Regressão Logística"
    assert contexto["recall"] == 0.9286
    assert contexto["falsos_negativos"] == 3


def test_contexto_de_gestao_falha_quando_o_modelo_escolhido_some():
    """Metrica ausente tem que estourar, nao virar None dentro do prompt."""
    with pytest.raises(KeyError):
        narrator.contexto_de_gestao({"teste": []})


def test_contexto_de_mudanca_le_o_summary_do_algoritmo_genetico():
    resumo = {
        "execucoes": 45,
        "bracos": [
            {"braco": "AG, só limiar", "custo_medio": 0.0791, "recall_medio": 0.9882,
             "features_medio": 30.0, "falsos_negativos_medio": 2.0},
            {"braco": "AG, completo", "custo_medio": 0.0872, "recall_medio": 0.9882,
             "features_medio": 4.3, "falsos_negativos_medio": 2.0},
        ],
    }

    contexto = narrator.contexto_de_mudanca(resumo)

    assert contexto["execucoes"] == 45
    assert contexto["melhor_braco"]["braco"] == "AG, só limiar"
    assert len(contexto["bracos"]) == 2

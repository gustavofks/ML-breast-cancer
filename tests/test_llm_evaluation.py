"""Testes dos verificadores automaticos de qualidade dos textos gerados.

Cada verificador e testado com um defeito injetado de proposito. Verificador que
nunca viu um defeito nao e verificador: e uma funcao que devolve `True`.
"""

import pytest

from src.llm import evaluation as aval

TEXTO_BOM = """
## Leitura do caso
A probabilidade estimada de malignidade é 0,91, acima do limiar de 0,25 adotado.

## Medidas que mais pesaram
As medidas que mais pesaram foram texture_worst, area_worst e concave points_mean.

## Ressalva
Esta leitura é apoio à decisão e não substitui a avaliação do médico responsável.
"""

FEATURES = ["texture_worst", "area_worst", "concave points_mean"]
SECOES = ["Leitura do caso", "Medidas que mais pesaram", "Ressalva"]


def test_fidelidade_aceita_texto_que_so_usa_os_valores_fornecidos():
    resultado = aval.verificar_fidelidade(TEXTO_BOM, valores=[0.91, 0.25])

    assert resultado.aprovado
    assert resultado.detalhes == []


def test_fidelidade_pega_numero_falso_injetado():
    """Alucinacao de estatistica: o erro mais grave desta camada."""
    adulterado = TEXTO_BOM.replace("0,91", "0,97")

    resultado = aval.verificar_fidelidade(adulterado, valores=[0.91, 0.25])

    assert not resultado.aprovado
    assert "0,97" in " ".join(resultado.detalhes)


def test_fidelidade_tolera_diferenca_de_arredondamento():
    """0,9100 e 0,91 sao o mesmo numero; reprovar isso seria falso positivo."""
    resultado = aval.verificar_fidelidade("A probabilidade é 0,9100.", valores=[0.91])

    assert resultado.aprovado


def test_fidelidade_ignora_numero_colado_em_nome_de_medida():
    """`concave points_mean` nao traz numero; mas nomes com digito existem."""
    resultado = aval.verificar_fidelidade(
        "A medida fractal_dimension_se2 contribuiu.", valores=[]
    )

    assert resultado.aprovado


def test_seguranca_exige_a_ressalva_obrigatoria():
    sem_ressalva = TEXTO_BOM.replace(
        "Esta leitura é apoio à decisão e não substitui a avaliação do médico responsável.",
        "",
    )

    resultado = aval.verificar_seguranca(sem_ressalva)

    assert not resultado.aprovado


def test_seguranca_pega_frase_diagnostica_injetada():
    """Texto que se passaria por laudo e falha de contrato, nao de estilo."""
    adulterado = TEXTO_BOM + "\nO diagnóstico é câncer maligno confirmado."

    resultado = aval.verificar_seguranca(adulterado)

    assert not resultado.aprovado
    assert resultado.detalhes


@pytest.mark.parametrize(
    "frase",
    [
        "O achado precisa ser confirmado por biópsia.",
        "A hipótese só pode ser descartada com exame adicional.",
        "O resultado deve ser confirmado pelo médico responsável.",
    ],
)
def test_seguranca_aceita_a_linguagem_cautelosa_que_o_projeto_quer(frase):
    """Reprovar "precisa ser confirmado por biópsia" puniria o texto certo.

    Palavra solta nao distingue a afirmacao diagnostica da recomendacao
    prudente. O verificador tem que reprovar "diagnóstico confirmado" e aprovar
    "precisa ser confirmado", senao empurra o modelo para longe da cautela.
    """
    texto = f"{frase} Esta leitura não substitui a avaliação do médico responsável."

    resultado = aval.verificar_seguranca(texto)

    assert resultado.aprovado, resultado.detalhes


@pytest.mark.parametrize(
    "frase",
    [
        "O diagnóstico é câncer maligno confirmado.",
        "Trata-se de um tumor maligno confirmado.",
        "A malignidade está descartada.",
        "É câncer, sem dúvida.",
    ],
)
def test_seguranca_reprova_afirmacao_diagnostica_em_qualquer_forma(frase):
    texto = f"{frase} Esta leitura não substitui a avaliação do médico responsável."

    resultado = aval.verificar_seguranca(texto)

    assert not resultado.aprovado


def test_cobertura_exige_as_tres_features_de_maior_contribuicao():
    resultado = aval.verificar_cobertura(TEXTO_BOM, features=FEATURES)

    assert resultado.aprovado


def test_cobertura_reprova_explicacao_generica():
    """Texto que ignora a evidencia do caso serve para qualquer caso."""
    generico = "O modelo indicou risco elevado. Procure o médico."

    resultado = aval.verificar_cobertura(generico, features=FEATURES)

    assert not resultado.aprovado
    assert len(resultado.detalhes) == 3


def test_cobertura_aceita_nome_com_espaco_no_lugar_do_underline():
    """O modelo tende a escrever "texture worst"; isso nao e omissao."""
    texto = "Pesaram texture worst, area worst e concave points mean."

    resultado = aval.verificar_cobertura(texto, features=FEATURES)

    assert resultado.aprovado


def test_formato_exige_as_secoes_e_respeita_o_limite():
    resultado = aval.verificar_formato(TEXTO_BOM, secoes=SECOES)

    assert resultado.aprovado


def test_formato_reprova_secao_ausente():
    sem_secao = TEXTO_BOM.replace("## Ressalva", "## Observação")

    resultado = aval.verificar_formato(sem_secao, secoes=SECOES)

    assert not resultado.aprovado


def test_formato_reprova_texto_acima_do_limite():
    longo = TEXTO_BOM + " palavra" * 500

    resultado = aval.verificar_formato(longo, secoes=SECOES, limite_de_palavras=50)

    assert not resultado.aprovado


def test_avaliacao_completa_agrega_os_quatro_verificadores():
    relatorio = aval.avaliar(TEXTO_BOM, valores=[0.91, 0.25], features=FEATURES, secoes=SECOES)

    assert set(relatorio.verificacoes) == {"fidelidade", "seguranca", "cobertura", "formato"}
    assert relatorio.aprovado


def test_avaliacao_completa_reprova_se_qualquer_verificador_reprovar():
    relatorio = aval.avaliar(
        TEXTO_BOM.replace("0,91", "0,97"),
        valores=[0.91, 0.25],
        features=FEATURES,
        secoes=SECOES,
    )

    assert not relatorio.aprovado
    assert not relatorio.verificacoes["fidelidade"].aprovado
    assert relatorio.verificacoes["seguranca"].aprovado


def test_relatorio_serializa_para_json():
    """A avaliacao entra no artefato ao lado do texto, nao num log a parte."""
    relatorio = aval.avaliar(TEXTO_BOM, valores=[0.91, 0.25], features=FEATURES, secoes=SECOES)

    payload = relatorio.para_dicionario()

    assert payload["aprovado"] is True
    assert payload["verificacoes"]["fidelidade"]["aprovado"] is True
    assert isinstance(payload["verificacoes"]["cobertura"]["detalhes"], list)


@pytest.mark.parametrize("vazio", ["", "   \n  "])
def test_texto_vazio_reprova_em_vez_de_passar_por_omissao(vazio):
    """Verificador que aprova texto vazio deixa passar a falha mais obvia."""
    relatorio = aval.avaliar(vazio, valores=[0.91], features=FEATURES, secoes=SECOES)

    assert not relatorio.aprovado

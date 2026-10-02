"""Testes dos templates de prompt e do versionamento."""

import pytest

from src.llm import prompts


def test_todo_template_tem_identificador_e_versao():
    """O relatorio cita qual prompt produziu qual texto; sem versao, nao da."""
    assert prompts.TEMPLATES

    for identificador, template in prompts.TEMPLATES.items():
        assert template.identificador == identificador
        assert template.versao >= 1
        assert template.sistema
        assert template.usuario


def test_os_tres_produtos_de_texto_do_design_existem():
    """Explicacao por paciente, resumo de gestao e nota de mudanca."""
    nomes = {template.nome for template in prompts.TEMPLATES.values()}

    assert nomes == {"explicacao_paciente", "resumo_gestao", "nota_de_mudanca"}


def test_existe_mais_de_uma_versao_de_ao_menos_um_prompt():
    """Comparar versoes e o que torna "aplicamos prompt engineering" verificavel."""
    por_nome: dict[str, list[int]] = {}
    for template in prompts.TEMPLATES.values():
        por_nome.setdefault(template.nome, []).append(template.versao)

    assert any(len(versoes) > 1 for versoes in por_nome.values())


@pytest.mark.parametrize("identificador", list(prompts.TEMPLATES))
def test_prompt_de_sistema_declara_a_fronteira_clinica(identificador):
    """Apoio a decisao, nunca diagnostico. A fronteira vive no proprio prompt."""
    sistema = prompts.TEMPLATES[identificador].sistema.lower()

    assert "apoio" in sistema
    assert "não substitui" in sistema


@pytest.mark.parametrize("identificador", list(prompts.TEMPLATES))
def test_prompt_proibe_inventar_numero(identificador):
    """Alucinacao de estatistica e o erro mais grave desta camada."""
    texto = prompts.TEMPLATES[identificador].usuario.lower()

    assert "apenas os valores fornecidos" in texto


@pytest.mark.parametrize("identificador", list(prompts.TEMPLATES))
def test_prompt_declara_o_contrato_de_saida(identificador):
    """Secoes e limite de tamanho; sem contrato, o verificador de formato nao tem o que conferir."""
    template = prompts.TEMPLATES[identificador]

    assert template.secoes
    for secao in template.secoes:
        assert secao in template.usuario


def test_evidencia_entra_como_json_e_nao_como_prosa():
    """Numero em prosa convida o modelo a reescrever; JSON convida a copiar."""
    template = prompts.TEMPLATES["explicacao_paciente_v2"]

    renderizado = template.renderizar({"probabilidade": 0.91, "limiar": 0.25})

    assert '"probabilidade": 0.91' in renderizado


def test_versao_2_traz_exemplo_resolvido_e_a_versao_1_nao():
    """A comparacao entre versoes precisa de versoes de fato diferentes."""
    v1 = prompts.TEMPLATES["explicacao_paciente_v1"]
    v2 = prompts.TEMPLATES["explicacao_paciente_v2"]

    assert "exemplo" not in v1.usuario.lower()
    assert "exemplo" in v2.usuario.lower()


def test_renderizacao_falha_quando_falta_evidencia():
    """Template que renderiza com buraco produz texto plausivel e errado."""
    template = prompts.TEMPLATES["explicacao_paciente_v2"]

    with pytest.raises(KeyError):
        template.renderizar({})


def test_versoes_do_mesmo_prompt_produzem_textos_diferentes():
    """Se as duas versoes renderizassem igual, a comparacao nao mediria nada."""
    evidencia = {"probabilidade": 0.91, "limiar": 0.25}

    v1 = prompts.TEMPLATES["explicacao_paciente_v1"].renderizar(evidencia)
    v2 = prompts.TEMPLATES["explicacao_paciente_v2"].renderizar(evidencia)

    assert v1 != v2


def test_versoes_de_um_prompt_sao_recuperaveis_pelo_nome():
    versoes = prompts.versoes_de("explicacao_paciente")

    assert [t.versao for t in versoes] == [1, 2]

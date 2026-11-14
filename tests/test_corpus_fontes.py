"""Testes das fontes e dos temas dos protocolos."""

from src import config
from src.corpus import fontes


def test_treze_temas_com_ids_unicos_e_papeis_validos():
    ids = [t.id for t in fontes.TEMAS]

    assert ids == [f"PROT-{i:02d}" for i in range(1, 14)]
    assert {t.papel for t in fontes.TEMAS} <= {"treino", "inedito"}
    assert [t.id for t in fontes.TEMAS if t.papel == "inedito"] == ["PROT-07"]


def test_todo_tema_externo_aponta_para_fonte_conhecida_com_paginas_validas():
    for tema in fontes.TEMAS:
        if tema.fonte == "interno":
            continue
        assert tema.fonte in fontes.FONTES
        for inicio, fim in tema.paginas:
            assert 1 <= inicio <= fim


def test_limpeza_tira_cabecalho_numero_de_pagina_e_junta_hifenizacao():
    sujo = ("23 Instituto Nacional de Câncer / MS\n"
            "Parâmetros Técnicos para Programação de Ações de Detecção Precoce\n"
            "A mamo-\ngrafia deve ser feita a cada dois anos.\n24\n")

    limpo = fontes.limpar(sujo)

    assert "Instituto Nacional" not in limpo and "Parâmetros Técnicos" not in limpo
    assert "mamografia deve ser feita a cada dois anos." in limpo
    assert "24" not in limpo.split()


def test_fatos_do_protocolo_interno_vem_da_configuracao():
    v1 = fontes.fatos_internos(versao=1)
    v2 = fontes.fatos_internos(versao=2)

    assert str(config.CLINICA_PRAZO_PENDENCIA_DIAS) in v1
    assert str(config.CLINICA_PRAZO_PENDENCIA_DIAS_V2) in v2
    assert str(config.CLINICA_PRAZO_PENDENCIA_DIAS) not in v2

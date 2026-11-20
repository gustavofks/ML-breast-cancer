"""Testes da busca nos protocolos vigentes."""

from src import config
from src.corpus import busca, protocolos

PROTOCOLOS = protocolos.carregar(config.CLINICA_PROTOCOLOS_DIR)


def test_so_a_ultima_versao_de_cada_protocolo_entra_no_indice():
    vigentes = busca.vigentes(PROTOCOLOS)
    ids = [p.id for p in vigentes]

    assert len(ids) == len(set(ids))
    for p in vigentes:
        assert p.versao == max(q.versao for q in PROTOCOLOS if q.id == p.id)
    assert any(p.id == "PROT-13" and p.versao == 2 for p in vigentes)


def test_o_texto_de_um_trecho_encontra_o_proprio_trecho():
    indice = busca.indice_padrao()
    for trecho in indice.trechos:
        achados = busca.melhores_trechos(indice, trecho.texto, k=4)
        assert trecho.id in [a.trecho.id for a in achados]


def test_modo_protocolo_devolve_todas_as_secoes_de_um_protocolo_so_em_ordem():
    indice = busca.indice_padrao()
    achados = busca.melhor_protocolo(indice, "rastreamento mamografico em mulheres de 50 a 69 anos")

    assert len({a.trecho.protocolo for a in achados}) == 1
    ids = [a.trecho.id for a in achados]
    assert ids == sorted(ids) and len(ids) == 4


def test_pergunta_sem_vocabulario_comum_nao_recupera_nada():
    indice = busca.indice_padrao()

    assert busca.melhores_trechos(indice, "xyzzy qwerty", k=4) == []
    assert busca.melhor_protocolo(indice, "xyzzy qwerty") == []


def test_abaixo_do_limiar_a_pergunta_e_geral_e_vai_sem_trechos():
    indice = busca.indice_padrao()
    pergunta = "rastreamento mamografico em mulheres de 50 a 69 anos"

    assert busca.recuperar(indice, pergunta, modo="protocolo", k=4, limiar=0.0)
    assert busca.recuperar(indice, pergunta, modo="protocolo", k=4, limiar=1e9) == []


def test_busca_e_deterministica():
    indice = busca.indice_padrao()
    pergunta = "prazo para exame pendente"

    assert busca.melhores_trechos(indice, pergunta, k=4) == busca.melhores_trechos(indice, pergunta, k=4)

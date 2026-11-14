"""Testes dos protocolos em markdown, trechos e citacoes."""

from src.corpus import protocolos

TEXTO = """# Autoexame das mamas

## Objetivo
Orientar a equipe.

## Conduta
1. Nao recomendar o autoexame como rastreamento.
2. Valorizar queixas da paciente.
"""


def _protocolo(versao=1):
    return protocolos.Protocolo(id="PROT-03", versao=versao, titulo="Autoexame das mamas",
                                fonte="diretrizes_2015 p. 75-80", papel="treino", texto=TEXTO)


def test_salvar_e_carregar_devolve_o_mesmo_protocolo(tmp_path):
    original = _protocolo()
    protocolos.salvar(original, tmp_path)

    assert protocolos.carregar(tmp_path) == [original]


def test_versoes_diferentes_sao_arquivos_diferentes(tmp_path):
    protocolos.salvar(_protocolo(1), tmp_path)
    protocolos.salvar(_protocolo(2), tmp_path)

    assert sorted(p.versao for p in protocolos.carregar(tmp_path)) == [1, 2]


def test_cada_secao_vira_um_trecho_com_id_estavel():
    trechos = protocolos.trechos(_protocolo())

    assert [t.id for t in trechos] == ["PROT-03 v1 §1", "PROT-03 v1 §2"]
    assert "autoexame como rastreamento" in trechos[1].texto


def test_citacoes_sao_lidas_do_texto_da_resposta():
    resposta = "Nao se recomenda [PROT-03 v1 §2], e a queixa importa [PROT-03 v1 §2] [PROT-11 v1 §1]."

    assert protocolos.citacoes_em(resposta) == ["PROT-03 v1 §2", "PROT-11 v1 §1"]


def test_bloco_de_trechos_usa_o_mesmo_formato_da_citacao():
    bloco = protocolos.bloco_de_trechos(protocolos.trechos(_protocolo()))

    assert bloco.startswith("[PROT-03 v1 §1]")
    assert "[PROT-03 v1 §2]" in bloco

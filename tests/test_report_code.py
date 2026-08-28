"""Testes da extracao e do realce dos trechos de codigo do relatorio.

O ponto que estes testes protegem: os trechos exibidos na pagina precisam ser
o codigo que roda de verdade. Se um deles puder divergir do arquivo-fonte, a
pagina passa a explicar um pipeline que nao existe mais.
"""

import pytest

from src.report_code import RAIZ, TrechoNaoEncontrado, bloco, destacar, fonte


def test_trecho_e_identico_ao_arquivo_fonte():
    codigo = fonte("src/preprocessing.py", "build_pipeline")
    original = (RAIZ / "src" / "preprocessing.py").read_text(encoding="utf-8")

    assert codigo.startswith("def build_pipeline(")
    # Cada linha de codigo do trecho existe, igual, no arquivo original.
    for linha in codigo.splitlines():
        if linha.strip():
            assert linha in original, linha


def test_docstring_sai_do_trecho():
    codigo = fonte("src/preprocessing.py", "build_pipeline")

    assert "Encadeia o pre-processamento" not in codigo
    assert '"""' not in codigo
    # O corpo continua inteiro.
    assert "build_preprocessor()" in codigo
    assert "estimator" in codigo


def test_varias_funcoes_saem_na_ordem_pedida():
    codigo = fonte("src/preprocessing.py", "split_data", "build_pipeline")
    assert codigo.index("def split_data") < codigo.index("def build_pipeline")


def test_nome_inexistente_quebra_a_geracao():
    # Falhar aqui e proposital: um trecho renomeado precisa quebrar a geracao
    # do relatorio, nao sumir em silencio da pagina.
    with pytest.raises(TrechoNaoEncontrado):
        fonte("src/preprocessing.py", "funcao_que_nao_existe")


def test_realce_marca_palavra_reservada_texto_e_comentario():
    marcado = destacar('def f():\n    # nota\n    return "oi"\n')

    assert '<span class="k">def</span>' in marcado
    assert '<span class="d">f</span>' in marcado
    assert '<span class="c"># nota</span>' in marcado
    assert '<span class="s">&quot;oi&quot;</span>' in marcado


def test_realce_escapa_html():
    marcado = destacar("x = a < b and c > d\n")

    assert "&lt;" in marcado and "&gt;" in marcado
    assert "<b>" not in marcado


def test_realce_preserva_a_indentacao():
    codigo = "def f():\n    if True:\n        return 1\n"
    marcado = destacar(codigo)

    # Removendo as marcacoes, o texto precisa voltar ao original.
    import re
    from html import unescape

    limpo = unescape(re.sub(r"</?span[^>]*>", "", marcado))
    assert limpo == codigo


def test_realce_degrada_sem_perder_o_codigo():
    # Codigo que nao tokeniza volta apenas escapado: perde a cor, nunca o texto.
    quebrado = 'def f(:\n    "aberta\n'
    assert "def f(:" in destacar(quebrado)


def test_bloco_entrega_html_pronto():
    marcado = bloco("src/evaluation.py", "threshold_analysis")
    assert '<span class="k">' in marcado
    assert "predict_proba" in marcado

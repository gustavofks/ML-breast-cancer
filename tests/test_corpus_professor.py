"""Testes dos pedidos ao modelo professor e da leitura das respostas."""

import pytest

from src.corpus import professor


def test_pedido_de_protocolo_proibe_acrescentar_fatos_e_traz_a_fonte():
    mensagens = professor.pedido_protocolo("Autoexame", "Fato A. Fato B.")

    texto = mensagens[-1]["content"]
    assert "Fato A. Fato B." in texto
    assert "SOMENTE" in texto
    assert "## Conduta" in texto


def test_lista_json_e_lida_mesmo_dentro_de_cerca_de_markdown():
    resposta = 'Aqui:\n```json\n[{"pergunta": "a?", "resposta": "b"}]\n```\nFim.'

    assert professor.ler_lista_json(resposta) == [{"pergunta": "a?", "resposta": "b"}]


def test_lista_json_quebrada_vira_lista_vazia_e_nao_excecao():
    assert professor.ler_lista_json("sem json nenhum") == []
    assert professor.ler_lista_json('[{"pergunta": "a?"') == []


def test_itens_sem_os_campos_pedidos_sao_descartados():
    resposta = '[{"pergunta": "a?", "resposta": "b"}, {"pergunta": "so pergunta"}]'

    itens = professor.ler_lista_json(resposta, campos=("pergunta", "resposta"))

    assert itens == [{"pergunta": "a?", "resposta": "b"}]


@pytest.mark.parametrize("fabrica", [
    lambda: professor.pedido_perguntas("TRECHOS", n=5, ids=["PROT-01 v1 §1"]),
    lambda: professor.pedido_recusas(5),
    lambda: professor.pedido_fora_do_escopo(5),
    lambda: professor.pedido_gerais(5),
    lambda: professor.pedido_verificacao("doc", "fonte"),
    lambda: professor.pedido_nao_encontrado("pergunta?", "trechos"),
])
def test_todo_pedido_e_uma_lista_de_mensagens_de_usuario(fabrica):
    mensagens = fabrica()

    assert mensagens and all(m["role"] == "user" and m["content"] for m in mensagens)


def test_pedido_de_perguntas_exige_citacao_com_os_ids_recebidos():
    texto = professor.pedido_perguntas("TRECHOS", n=5, ids=["PROT-01 v1 §1", "PROT-01 v1 §2"])[-1]["content"]

    assert "[PROT-01 v1 §1]" in texto and "[PROT-01 v1 §2]" in texto

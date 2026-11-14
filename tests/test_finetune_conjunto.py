"""Testes da montagem dos conjuntos de treino, validacao e teste."""

import pytest

from src import config
from src.finetune import conjunto

ORIGEM = config.FASE3_ORIGEM_PROFESSOR


def _ex(pergunta, protocolo=None, tipo="pergunta"):
    return conjunto.Exemplo(tipo=tipo, protocolo=protocolo, origem=ORIGEM,
                            mensagens=[{"role": "user", "content": f"PERGUNTA: {pergunta}"},
                                       {"role": "assistant", "content": "resposta"}])


def test_exemplo_de_origem_nao_permitida_e_recusado():
    with pytest.raises(ValueError):
        conjunto.Exemplo(tipo="pergunta", protocolo=None, origem="claude-haiku-5-5",
                         mensagens=[{"role": "user", "content": "x"},
                                    {"role": "assistant", "content": "y"}])


def test_exemplo_de_treino_nao_tem_mensagem_de_sistema():
    with pytest.raises(ValueError):
        conjunto.Exemplo(tipo="pergunta", protocolo=None, origem=ORIGEM,
                         mensagens=[{"role": "system", "content": "instrucao"},
                                    {"role": "user", "content": "x"},
                                    {"role": "assistant", "content": "y"}])


def test_mensagem_com_trechos_usa_rotulos_de_formato():
    mensagem = conjunto.mensagem_do_usuario("Qual a idade?", "[PROT-01 v1 §1]\n## Conduta\n...")

    assert mensagem.startswith("TRECHOS:\n[PROT-01 v1 §1]")
    assert mensagem.rstrip().endswith("PERGUNTA: Qual a idade?")


def test_protocolo_inedito_nunca_aparece_no_treino_nem_na_validacao():
    treino = [_ex(f"pergunta {i} sobre tema {i}", "PROT-01") for i in range(20)]
    teste = [_ex("pergunta sobre termografia", "PROT-07")]

    divisoes = conjunto.dividir(treino, teste, ineditos={"PROT-07"}, semente=42)

    for nome in ("treino", "validacao"):
        assert all(e.protocolo != "PROT-07" for e in divisoes[nome])
    assert any(e.protocolo == "PROT-07" for e in divisoes["teste"])


def test_pergunta_quase_igual_a_uma_do_teste_sai_do_treino():
    treino = [_ex("Qual a idade do rastreamento?", "PROT-01"),
              _ex("Quando fazer ultrassom no rastreamento?", "PROT-06")]
    teste = [_ex("Qual é a idade do rastreamento?", "PROT-01")]

    divisoes = conjunto.dividir(treino, teste, ineditos=set(), semente=42)

    perguntas = [e.pergunta for e in divisoes["treino"] + divisoes["validacao"]]
    assert "Qual a idade do rastreamento?" not in perguntas
    assert divisoes["vazamentos_removidos"] == 1


def test_separacao_e_deterministica_pela_semente():
    treino = [_ex(f"pergunta numero {i}", "PROT-01") for i in range(30)]

    a = conjunto.dividir(treino, [], ineditos=set(), semente=42)
    b = conjunto.dividir(treino, [], ineditos=set(), semente=42)

    assert [e.pergunta for e in a["validacao"]] == [e.pergunta for e in b["validacao"]]


def test_jsonl_ida_e_volta(tmp_path):
    exemplos = [_ex("pergunta 1", "PROT-01"), _ex("pergunta 2", None, "geral")]
    caminho = tmp_path / "x.jsonl"

    conjunto.salvar_jsonl(exemplos, caminho)

    assert conjunto.carregar_jsonl(caminho) == exemplos

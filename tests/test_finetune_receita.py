"""Testes da receita de treino e da conversao dos exemplos."""

import pytest

from src import config
from src.finetune import receita


def _renderizar(mensagens, *, gerar):
    """Molde de chat falso, no formato do Qwen."""
    texto = "".join(f"<|{m['role']}|>{m['content']}<|fim|>" for m in mensagens)
    return texto + ("<|assistant|>" if gerar else "")


def _renderizar_qwen(mensagens, *, gerar):
    """Como o molde real: o turno do assistente abre com o mesmo marcador."""
    texto = "".join(f"<|{m['role']}|>{m['content']}<|fim|>" if m["role"] != "assistant"
                    else f"<|assistant|>{m['content']}<|fim|>" for m in mensagens)
    return texto + ("<|assistant|>" if gerar else "")


EXEMPLO = {"tipo": "pergunta", "protocolo": "PROT-01", "origem": config.FASE3_ORIGEM_PROFESSOR,
           "mensagens": [{"role": "user", "content": "PERGUNTA: Qual a idade?"},
                         {"role": "assistant", "content": "Aos 50 anos [PROT-01 v1 §2]."}]}


def test_prompt_termina_onde_a_resposta_comeca():
    par = receita.para_prompt_completion(EXEMPLO, _renderizar_qwen)

    assert par["prompt"].endswith("<|assistant|>")
    assert par["completion"].startswith("Aos 50 anos")
    assert par["prompt"] + par["completion"] == _renderizar_qwen(EXEMPLO["mensagens"], gerar=False)


def test_resposta_nao_aparece_no_prompt():
    par = receita.para_prompt_completion(EXEMPLO, _renderizar_qwen)

    assert "Aos 50 anos" not in par["prompt"]


def test_molde_que_nao_prolonga_o_prompt_e_erro_e_nao_treino_silencioso():
    def molde_incoerente(mensagens, *, gerar):
        return "outro texto" if gerar else "texto completo"

    with pytest.raises(ValueError):
        receita.para_prompt_completion(EXEMPLO, molde_incoerente)


def test_passos_por_epoca_arredondam_para_cima():
    assert receita.passos_por_epoca(398, acumulacao=8) == 50
    assert receita.passos_por_epoca(8, acumulacao=8) == 1


def test_receita_padrao_vem_da_configuracao():
    r = receita.receita_padrao()

    assert r.modelo_base == config.FASE3_MODELO_BASE
    assert r.max_tokens == config.FASE3_TREINO_MAX_TOKENS
    assert r.lora_r == config.FASE3_LORA_R

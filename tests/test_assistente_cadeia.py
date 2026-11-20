"""Testes da cadeia com modelo falso: formato da mensagem, guardas e fecho."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import config
from src.assistente import cadeia
from src.corpus import busca, protocolos
from src.finetune import conjunto

PERGUNTA = "rastreamento mamografico em mulheres de 50 a 69 anos"


def _modelo(resposta, vistos, finish="stop"):
    """Modelo falso: guarda as mensagens recebidas e devolve um AIMessage."""
    def responder(prompt):
        vistos.append(prompt.to_messages())
        texto = resposta(prompt.to_messages()) if callable(resposta) else resposta
        return AIMessage(content=texto, response_metadata={"finish_reason": finish})
    return RunnableLambda(responder)


def _cita_o_primeiro(mensagens):
    """Cita o primeiro trecho recebido, com uma frase apoiada nele."""
    conteudo = mensagens[-1].content
    primeiro = conteudo.split("[", 1)[1].split("]", 1)[0]
    texto = conteudo.split(f"[{primeiro}]\n", 1)[1].split("\n\n", 1)[0]
    frase = " ".join(texto.replace("#", " ").split()[:12])
    return f"{frase} [{primeiro}]."


def _assistente(resposta, vistos, instrucoes=None):
    return cadeia.Assistente(_modelo(resposta, vistos), busca.indice_padrao(), limiar=0.0,
                             instrucoes=instrucoes)


def test_mensagem_sem_paciente_e_identica_ao_formato_do_treino():
    trechos = busca.recuperar(busca.indice_padrao(), PERGUNTA, modo="trechos", k=4, limiar=0.0)
    esperado = conjunto.mensagem_do_usuario(PERGUNTA, protocolos.bloco_de_trechos(trechos))

    assert cadeia.mensagem(PERGUNTA, trechos) == esperado


def test_resposta_citada_passa_e_ganha_o_fecho():
    vistos = []
    r = _assistente(_cita_o_primeiro, vistos).responder(PERGUNTA)

    assert r.verificacao.aprovado, r.verificacao.motivos
    assert r.modo == "protocolo"
    assert r.texto_final.endswith(config.FASE3_FECHO)


def test_sem_instrucoes_nao_ha_mensagem_de_sistema():
    vistos = []
    _assistente(_cita_o_primeiro, vistos).responder(PERGUNTA)

    assert [m.type for m in vistos[0]] == ["human"]


def test_com_instrucoes_a_primeira_mensagem_e_de_sistema():
    vistos = []
    _assistente(_cita_o_primeiro, vistos, instrucoes="regras").responder(PERGUNTA)

    assert [m.type for m in vistos[0]] == ["system", "human"]
    assert vistos[0][0].content == "regras"


def test_citacao_inventada_retem_a_resposta():
    vistos = []
    r = _assistente("Rastreamento bienal [PROT-99 v1 §1].", vistos).responder(PERGUNTA)

    assert not r.verificacao.aprovado
    assert "retida" in r.texto_final and "PROT-99" in r.texto_final
    assert "Rastreamento bienal" not in r.texto_final


def test_pergunta_geral_vai_sem_trechos_e_com_aviso():
    vistos = []
    r = cadeia.Assistente(_modelo("Os probióticos equilibram a flora intestinal.", vistos),
                          busca.indice_padrao(), limiar=1e9).responder("O que são probióticos?")

    assert r.modo == "geral" and r.trechos == ()
    assert "TRECHOS" not in vistos[0][-1].content
    assert config.FASE3_AVISO_GERAL in r.texto_final


def test_dados_da_paciente_vem_rotulados_como_registro_antes_dos_trechos():
    vistos = []
    _assistente(_cita_o_primeiro, vistos).responder(PERGUNTA, contexto_paciente="Paciente: PACIENTE-0001")
    conteudo = vistos[0][-1].content

    assert conteudo.startswith("DADOS DA PACIENTE")
    assert conteudo.index("PACIENTE-0001") < conteudo.index("TRECHOS:")


def test_chaves_no_texto_da_paciente_nao_quebram_o_prompt():
    vistos = []
    _assistente(_cita_o_primeiro, vistos).responder(PERGUNTA, contexto_paciente="medidas {area_mean}")

    assert "{area_mean}" in vistos[0][-1].content


def test_resposta_cortada_e_retida():
    vistos = []
    modelo = _modelo(_cita_o_primeiro, vistos, finish="length")  # resposta cortada pelo servidor
    r = cadeia.Assistente(modelo, busca.indice_padrao(), limiar=0.0).responder(PERGUNTA)

    assert not r.verificacao.aprovado

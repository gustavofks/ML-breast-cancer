"""Testes do fluxo: pausa para o medico, retomada, reprovacao e alertas."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver

from src.api.servico import Previsao
from src.assistente import cadeia, fluxo
from src.assistente.auditoria import Auditoria
from src.clinica import banco
from src.corpus import busca

PAAF = {"area_mean": 1.0}
REGISTROS = [
    {"tipo": "paciente", "paciente": "PACIENTE-0001", "idade": 52, "historico": "h", "origem": "t"},
    {"tipo": "paciente", "paciente": "PACIENTE-0002", "idade": 61, "historico": "h", "origem": "t"},
    {"tipo": "exame", "paciente": "PACIENTE-0001", "exame": "PAAF", "solicitado_em": "2026-10-20",
     "resultado_em": None, "resultado": None, "medidas": PAAF, "origem": "t"},
    {"tipo": "exame", "paciente": "PACIENTE-0002", "exame": "Mamografia", "solicitado_em": "2026-08-01",
     "resultado_em": None, "resultado": None, "medidas": None, "origem": "t"},
]


def _cita_o_primeiro(prompt):
    conteudo = prompt.to_messages()[-1].content
    if "TRECHOS:" not in conteudo:
        return "Sem protocolo para este caso."
    bloco = conteudo.split("TRECHOS:\n", 1)[1]
    primeiro = bloco.split("[", 1)[1].split("]", 1)[0]
    texto = bloco.split(f"[{primeiro}]\n", 1)[1].split("\n\n", 1)[0]
    return " ".join(texto.replace("#", " ").split()[:12]) + f" [{primeiro}]."


def _modelo(resposta):
    def responder(prompt):
        texto = resposta(prompt) if callable(resposta) else resposta
        return AIMessage(content=texto, response_metadata={"finish_reason": "stop"})
    return RunnableLambda(responder)


def _classificar(probabilidade):
    return lambda medidas: Previsao(probabilidade=probabilidade, limiar=0.5,
                                    classe="Maligno" if probabilidade >= 0.5 else "Benigno")


@pytest.fixture
def ambiente(tmp_path):
    caminho = tmp_path / "p.sqlite"
    banco.criar(caminho, REGISTROS)
    abertos = []

    def montar(resposta=_cita_o_primeiro, probabilidade=0.9):
        con = banco.conectar(caminho)
        pausa = sqlite3.connect(tmp_path / "c.sqlite", check_same_thread=False)
        abertos.extend([con, pausa])
        dep = fluxo.Dependencias(
            con=con, assistente=cadeia.Assistente(_modelo(resposta), busca.indice_padrao(), limiar=0.0),
            auditoria=Auditoria(tmp_path / "a.jsonl"), classificar=_classificar(probabilidade),
            versao={"modelo": "falso"})
        return fluxo.Fluxo(dep, SqliteSaver(pausa)), dep.auditoria

    yield montar
    for c in abertos:
        c.close()


def _nos(auditoria, execucao):
    return [e["no"] for e in auditoria.ler(execucao)]


def test_sugestao_aprovada_para_esperando_o_medico(ambiente):
    f, auditoria = ambiente()
    s = f.iniciar("PACIENTE-0001")

    assert s["status"] == "aguardando_medico"
    assert s["verificacao"]["aprovado"] and s["sugestao"]
    assert "registrar" not in _nos(auditoria, s["execucao"])


def test_decisao_do_medico_fecha_o_fluxo_e_fica_na_auditoria(ambiente):
    f, auditoria = ambiente()
    s = f.iniciar("PACIENTE-0001")
    final = f.decidir(s["execucao"], {"acao": "aprovar", "medico": "dra-teste"})
    nos = _nos(auditoria, s["execucao"])

    assert final["status"] == "concluido" and final["decisao"]["acao"] == "aprovar"
    assert nos.index("aprovacao_medica") < nos.index("registrar")
    assert nos.count("aprovacao_medica") == 1


def test_sugestao_reprovada_vira_alerta_e_nao_chega_ao_medico(ambiente):
    f, auditoria = ambiente(resposta="Conduta inventada [PROT-99 v1 §1].")
    s = f.iniciar("PACIENTE-0001")

    assert s["status"] == "concluido" and s["decisao"] is None
    assert any(a.startswith("Sugestão retida") for a in s["alertas"])
    assert "aprovacao_medica" not in _nos(auditoria, s["execucao"])


def test_pausa_sobrevive_a_reinicio_sem_repetir_passos(ambiente):
    f, auditoria = ambiente()
    s = f.iniciar("PACIENTE-0001")
    depois_do_reinicio, _ = ambiente()  # nova conexao, novo grafo, mesmos arquivos

    assert depois_do_reinicio.situacao(s["execucao"])["status"] == "aguardando_medico"
    final = depois_do_reinicio.decidir(s["execucao"], {"acao": "rejeitar", "medico": "dra-teste"})
    assert final["status"] == "concluido"
    assert _nos(auditoria, s["execucao"]).count("receber") == 1


def test_nada_aprovado_chega_a_registrar_sem_o_medico(ambiente):
    for resposta in (_cita_o_primeiro, "Conduta inventada [PROT-99 v1 §1]."):
        for paciente in ("PACIENTE-0001", "PACIENTE-0002"):
            f, auditoria = ambiente(resposta=resposta)
            s = f.iniciar(paciente)
            if s["status"] == "aguardando_medico":
                s = f.decidir(s["execucao"], {"acao": "aprovar", "medico": "dra-teste"})
            nos = _nos(auditoria, s["execucao"])
            if s["verificacao"]["aprovado"]:
                assert "aprovacao_medica" in nos and nos.index("aprovacao_medica") < nos.index("registrar")


def test_pendencia_vencida_e_caso_perto_do_limiar_viram_alerta(ambiente):
    f, _ = ambiente(probabilidade=0.55)
    pendente = f.iniciar("PACIENTE-0002")
    perto = f.iniciar("PACIENTE-0001")

    assert any(a.startswith("Pendência vencida: Mamografia") for a in pendente["alertas"])
    assert any(a.startswith("Caso perto do limiar") for a in perto["alertas"])


def test_classificacao_so_quando_ha_paaf(ambiente):
    f, _ = ambiente()

    assert f.iniciar("PACIENTE-0001")["classificacao"]["classe"] == "Maligno"
    assert f.iniciar("PACIENTE-0002")["classificacao"] is None


def test_decisao_invalida_e_recusada_e_o_fluxo_continua_esperando(ambiente):
    f, _ = ambiente()
    s = f.iniciar("PACIENTE-0001")

    with pytest.raises(ValueError):
        f.decidir(s["execucao"], {"acao": "prescrever", "medico": "dra-teste"})
    with pytest.raises(ValueError):
        f.decidir(s["execucao"], {"acao": "editar", "medico": "dra-teste"})  # editar exige texto
    assert f.situacao(s["execucao"])["status"] == "aguardando_medico"


def test_decidir_execucao_concluida_e_erro(ambiente):
    f, _ = ambiente()
    s = f.iniciar("PACIENTE-0001")
    f.decidir(s["execucao"], {"acao": "aprovar", "medico": "dra-teste"})

    with pytest.raises(fluxo.EstadoInvalido):
        f.decidir(s["execucao"], {"acao": "aprovar", "medico": "dra-teste"})


def test_paciente_inexistente_e_erro_claro(ambiente):
    f, _ = ambiente()

    with pytest.raises(LookupError):
        f.iniciar("PACIENTE-9999")
    with pytest.raises(LookupError):
        f.situacao("nao-existe")


def test_pergunta_do_caso_com_pendencia_acha_o_protocolo_de_pendencias():
    pergunta = fluxo.pergunta_do_caso(None, [{"exame": "Mamografia", "solicitado_em": "2026-08-01"}])
    trechos = busca.recuperar(busca.indice_padrao(), pergunta, modo="trechos", k=4,
                              limiar=fluxo.config.FASE3_BUSCA_LIMIAR)

    assert any(t.protocolo == "PROT-13" for t in trechos)

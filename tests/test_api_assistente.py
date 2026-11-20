"""Testes das rotas do assistente, com modelo e classificador falsos."""

import sqlite3

import pytest

pytest.importorskip("fastapi", reason="requer requirements-api.txt")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from src.api import assistente as rotas  # noqa: E402
from src.api import main, servico  # noqa: E402
from src.api.servico import Previsao  # noqa: E402
from src.assistente import cadeia, fluxo  # noqa: E402
from src.assistente.auditoria import Auditoria  # noqa: E402
from src.clinica import banco  # noqa: E402
from src.corpus import busca  # noqa: E402

REGISTROS = [
    {"tipo": "paciente", "paciente": "PACIENTE-0001", "idade": 52, "historico": "h", "origem": "t"},
    {"tipo": "exame", "paciente": "PACIENTE-0001", "exame": "PAAF", "solicitado_em": "2026-10-20",
     "resultado_em": None, "resultado": None, "medidas": {"area_mean": 1.0}, "origem": "t"},
]


def _cita_o_primeiro(prompt):
    conteudo = prompt.to_messages()[-1].content
    if "TRECHOS:" not in conteudo:
        return "Os probióticos equilibram a flora intestinal."
    bloco = conteudo.split("TRECHOS:\n", 1)[1]
    primeiro = bloco.split("[", 1)[1].split("]", 1)[0]
    texto = bloco.split(f"[{primeiro}]\n", 1)[1].split("\n\n", 1)[0]
    return " ".join(texto.replace("#", " ").split()[:12]) + f" [{primeiro}]."


def _fora_do_ar(prompt):
    raise ConnectionError("servidor do modelo fora do ar")


@pytest.fixture(scope="module")
def modelo_fase2():
    return servico.Servico.construir(servico.referencia_da_fase_1())


@pytest.fixture
def montar(tmp_path, modelo_fase2):
    caminho = tmp_path / "p.sqlite"
    banco.criar(caminho, REGISTROS)
    abertos = []

    def _montar(resposta=_cita_o_primeiro):
        con = banco.conectar(caminho)
        pausa = sqlite3.connect(tmp_path / "c.sqlite", check_same_thread=False)
        abertos.extend([con, pausa])
        modelo = RunnableLambda(lambda p: AIMessage(content=resposta(p),
                                                    response_metadata={"finish_reason": "stop"}))
        assistente = cadeia.Assistente(modelo, busca.indice_padrao(), limiar=0.0)
        auditoria = Auditoria(tmp_path / "a.jsonl")
        dep = fluxo.Dependencias(con=con, assistente=assistente, auditoria=auditoria,
                                 classificar=lambda m: Previsao(0.9, 0.5, "Maligno"),
                                 versao={"modelo": "falso"})
        servico_assistente = rotas.ServicoAssistente(con=con, assistente=assistente,
                                                     fluxo=fluxo.Fluxo(dep, SqliteSaver(pausa)),
                                                     auditoria=auditoria)
        app = main.criar_app(modelo_fase2, diretorio_cache=tmp_path / "cache",
                             assistente=servico_assistente)
        return TestClient(app)

    yield _montar
    for c in abertos:
        c.close()


def test_pergunta_livre_responde_com_trechos_fecho_e_auditoria(montar):
    cliente = montar()
    r = cliente.post("/assistente/perguntar", json={"pergunta": "rastreamento mamografico 50 a 69 anos"})
    corpo = r.json()

    assert r.status_code == 200 and corpo["aprovado"]
    assert corpo["trechos"] and corpo["resposta"].endswith("médico responsável.")
    assert cliente.get(f"/auditoria/{corpo['execucao']}").json()["eventos"][0]["no"] == "pergunta_livre"


def test_fluxo_pausa_e_a_decisao_conclui(montar):
    cliente = montar()
    inicio = cliente.post("/fluxo", json={"paciente": "PACIENTE-0001"}).json()

    assert inicio["status"] == "aguardando_medico"
    fim = cliente.post(f"/fluxo/{inicio['execucao']}/decisao",
                       json={"acao": "aprovar", "medico": "dra-teste"}).json()
    assert fim["status"] == "concluido"
    nos = [e["no"] for e in cliente.get(f"/auditoria/{inicio['execucao']}").json()["eventos"]]
    assert nos[-2:] == ["aprovacao_medica", "registrar"]


def test_decisao_invalida_e_422_e_concluida_e_409(montar):
    cliente = montar()
    execucao = cliente.post("/fluxo", json={"paciente": "PACIENTE-0001"}).json()["execucao"]

    assert cliente.post(f"/fluxo/{execucao}/decisao", json={"acao": "prescrever", "medico": "d"}).status_code == 422
    assert cliente.post(f"/fluxo/{execucao}/decisao", json={"acao": "aprovar", "medico": "d"}).status_code == 200
    assert cliente.post(f"/fluxo/{execucao}/decisao", json={"acao": "aprovar", "medico": "d"}).status_code == 409


def test_paciente_ou_execucao_inexistente_e_404(montar):
    cliente = montar()

    assert cliente.post("/fluxo", json={"paciente": "PACIENTE-9999"}).status_code == 404
    assert cliente.post("/assistente/perguntar",
                        json={"pergunta": "exames?", "paciente": "PACIENTE-9999"}).status_code == 404
    assert cliente.post("/fluxo/nao-existe/decisao", json={"acao": "aprovar", "medico": "d"}).status_code == 404
    assert cliente.get("/auditoria/nao-existe").status_code == 404


def test_modelo_fora_do_ar_e_503_sem_resposta_inventada(montar):
    cliente = montar(resposta=_fora_do_ar)
    r = cliente.post("/assistente/perguntar", json={"pergunta": "rastreamento mamografico"})

    assert r.status_code == 503 and "indisponível" in r.json()["detail"]
    assert cliente.post("/fluxo", json={"paciente": "PACIENTE-0001"}).status_code == 503


def test_sem_assistente_as_rotas_nao_existem(modelo_fase2, tmp_path):
    cliente = TestClient(main.criar_app(modelo_fase2, diretorio_cache=tmp_path / "cache"))

    assert cliente.post("/assistente/perguntar", json={"pergunta": "x"}).status_code == 404

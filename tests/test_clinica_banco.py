"""Testes do banco de prontuarios."""

import sqlite3

import pytest

from src.clinica import banco

REGISTROS = [
    {"tipo": "paciente", "paciente": "PACIENTE-0001", "idade": 50, "historico": "h1", "origem": "t"},
    {"tipo": "paciente", "paciente": "PACIENTE-0002", "idade": 60, "historico": "h2", "origem": "t"},
    {"tipo": "exame", "paciente": "PACIENTE-0001", "exame": "Mamografia",
     "solicitado_em": "2026-08-01", "resultado_em": None, "resultado": None, "medidas": None, "origem": "t"},
    {"tipo": "exame", "paciente": "PACIENTE-0001", "exame": "PAAF",
     "solicitado_em": "2026-10-25", "resultado_em": None, "resultado": None,
     "medidas": {"area_mean": 1.5}, "origem": "t"},
    {"tipo": "exame", "paciente": "PACIENTE-0002", "exame": "Mamografia",
     "solicitado_em": "2026-07-01", "resultado_em": "2026-07-10", "resultado": "BI-RADS 2",
     "medidas": None, "origem": "t"},
    {"tipo": "evolucao", "paciente": "PACIENTE-0002", "data": "2026-10-01", "texto": "ev", "origem": "t"},
]


@pytest.fixture
def conexao(tmp_path):
    caminho = tmp_path / "p.sqlite"
    banco.criar(caminho, REGISTROS)
    con = banco.conectar(caminho)
    yield con
    con.close()


def test_consultas_nunca_devolvem_dados_de_outra_paciente(conexao):
    for consulta in (banco.exames, banco.evolucoes):
        for linha in consulta(conexao, "PACIENTE-0001"):
            assert linha["paciente"] == "PACIENTE-0001"
    assert banco.dados_paciente(conexao, "PACIENTE-0001")["idade"] == 50


def test_paciente_inexistente_devolve_vazio_e_nao_erro(conexao):
    assert banco.dados_paciente(conexao, "PACIENTE-9999") is None
    assert banco.exames(conexao, "PACIENTE-9999") == []


def test_pendencia_e_exame_sem_resultado_alem_do_prazo(conexao):
    pendentes = banco.pendencias(conexao, "PACIENTE-0001", hoje="2026-11-01", prazo_dias=30)

    assert [p["exame"] for p in pendentes] == ["Mamografia"]  # a PAAF de 7 dias nao venceu


def test_medidas_voltam_como_dicionario(conexao):
    paaf = [e for e in banco.exames(conexao, "PACIENTE-0001") if e["exame"] == "PAAF"][0]

    assert paaf["medidas"] == {"area_mean": 1.5}


def test_conexao_do_assistente_nao_consegue_escrever(conexao):
    with pytest.raises(sqlite3.OperationalError):
        conexao.execute("DELETE FROM exames")


def test_entrada_maliciosa_no_id_nao_vira_sql(conexao):
    assert banco.exames(conexao, "PACIENTE-0001' OR '1'='1") == []

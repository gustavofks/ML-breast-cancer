"""Testes da auditoria: um evento por linha, por execucao, sem texto do prontuario."""

import json

import pytest

from src.assistente.auditoria import Auditoria


def _auditoria(tmp_path):
    return Auditoria(tmp_path / "a.jsonl", relogio=lambda: "2026-11-20T10:00:00+00:00")


def test_evento_tem_execucao_no_e_horario(tmp_path):
    a = _auditoria(tmp_path)
    evento = a.registrar("x1", "receber", paciente="PACIENTE-0001")

    assert evento == {"execucao": "x1", "no": "receber", "horario": "2026-11-20T10:00:00+00:00",
                      "paciente": "PACIENTE-0001"}


def test_cada_evento_e_uma_linha_json(tmp_path):
    a = _auditoria(tmp_path)
    a.registrar("x1", "receber", paciente="PACIENTE-0001")
    a.registrar("x1", "verificar", aprovado=True, motivos=[])

    linhas = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(linha)["no"] for linha in linhas] == ["receber", "verificar"]


def test_ler_devolve_so_a_execucao_pedida_na_ordem(tmp_path):
    a = _auditoria(tmp_path)
    a.registrar("x1", "receber", paciente="PACIENTE-0001")
    a.registrar("x2", "receber", paciente="PACIENTE-0002")
    a.registrar("x1", "registrar", desfecho="aprovar")

    assert [e["no"] for e in a.ler("x1")] == ["receber", "registrar"]
    assert a.ler("x9") == []


@pytest.mark.parametrize("campo", ["contexto", "contexto_paciente", "historico", "evolucoes"])
def test_texto_do_prontuario_nao_entra_na_auditoria(tmp_path, campo):
    with pytest.raises(ValueError):
        _auditoria(tmp_path).registrar("x1", "consultar_prontuario", **{campo: "texto"})


def test_paciente_so_entra_como_pseudonimo(tmp_path):
    with pytest.raises(ValueError):
        _auditoria(tmp_path).registrar("x1", "receber", paciente="Maria da Silva")

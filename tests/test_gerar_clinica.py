"""Testes do script que monta a clinica ficticia."""

import importlib.util
import json

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "gerar_clinica.py"
    spec = importlib.util.spec_from_file_location("gerar_clinica", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_monta_os_quatro_artefatos_em_destino_isolado(script, tmp_path):
    script.montar(destino=tmp_path, n_pacientes=120)

    assert (tmp_path / "prontuarios.jsonl").exists()
    assert (tmp_path / "gabarito_paaf.json").exists()
    assert (tmp_path / "prontuarios.sqlite").exists()
    assert (tmp_path / "bruto" / "clinica.json").exists()


def test_gabarito_liga_os_114_casos_a_pseudonimos_existentes(script, tmp_path):
    script.montar(destino=tmp_path, n_pacientes=120)
    gabarito = json.loads((tmp_path / "gabarito_paaf.json").read_text(encoding="utf-8"))
    pacientes = {json.loads(linha)["paciente"]
                 for linha in (tmp_path / "prontuarios.jsonl").read_text(encoding="utf-8").splitlines()}

    assert sorted(g["caso"] for g in gabarito) == list(range(114))
    assert all(g["paciente"] in pacientes for g in gabarito)
    assert all(g["referencia"] in config.CLASS_NAMES for g in gabarito)


def test_duas_execucoes_gravam_o_mesmo_jsonl(script, tmp_path):
    script.montar(destino=tmp_path / "a", n_pacientes=120)
    script.montar(destino=tmp_path / "b", n_pacientes=120)

    a = (tmp_path / "a" / "prontuarios.jsonl").read_bytes()
    assert a == (tmp_path / "b" / "prontuarios.jsonl").read_bytes()

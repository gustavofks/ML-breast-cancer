"""Filmes do hospital: roteiros fieis aos artefatos e ligados ao que a cena sabe fazer."""

from __future__ import annotations

import json
import re

import pytest

from src import config
from src.ui import filmes, hospital

CASOS = [
    {"diagnostico_registrado": "Maligno", "probabilidade": 0.9},
    {"diagnostico_registrado": "Maligno", "probabilidade": 0.3},
    {"diagnostico_registrado": "Maligno", "probabilidade": 0.1},
    {"diagnostico_registrado": "Benigno", "probabilidade": 0.4},
    {"diagnostico_registrado": "Benigno", "probabilidade": 0.01},
]


def _vitrine() -> dict:
    caso = {
        "diagnostico_registrado": "Maligno",
        "classe": "Benigno",
        "probabilidade": 0.1,
        "limiar": 0.2,
        "medidas": {"texture_worst": 1.0},
        "contribuicoes": [{"feature": "texture_worst", "contribuicao": -1.0}],
        "texto": "Texto.",
        "texto_motivo": None,
    }
    return {"casos": [caso], "console": {"gravacao": {"eventos": []}}}


@pytest.fixture(scope="module")
def lista() -> list[dict]:
    return hospital.montar_dados(_vitrine())["filmes"]


def _planos(lista: list[dict]) -> list[dict]:
    return [plano for filme in lista for plano in filme["planos"]]


def _por_id(lista: list[dict], filme: str, plano: str) -> dict:
    alvo = next(f for f in lista if f["id"] == filme)
    return next(p for p in alvo["planos"] if p["id"] == plano)


def test_erros_no_corte_trocam_negativos_por_positivos():
    assert filmes.erros_no_corte(CASOS, 0.5) == (2, 0)
    assert filmes.erros_no_corte(CASOS, 0.2) == (1, 1)
    assert filmes.erros_no_corte(CASOS, 0.05) == (0, 1)


def test_um_filme_por_fase_e_cada_um_ergue_a_propria_ala(lista):
    assert [f["id"] for f in lista] == ["fase1", "fase2", "fase3"]
    construidas = []
    for filme in lista:
        assert filme["alas_prontas"] == construidas
        erguidas = [
            acao.split(":")[1]
            for plano in filme["planos"]
            for acao in plano["acoes"]
            if acao.startswith("construir:")
        ]
        assert erguidas
        construidas = construidas + erguidas


def test_o_dia_avanca_de_um_filme_para_o_outro(lista):
    fim_anterior = 0
    for filme in lista:
        assert filme["hora"][0] >= fim_anterior
        fim_anterior = filme["hora"][1]


def test_tomadas_citadas_existem_na_cena(lista):
    fonte = (config.HOSPITAL_DIR / "tomadas.js").read_text(encoding="utf-8")
    existentes = set(re.findall(r"^  (\w+): \{", fonte, re.M))
    assert {plano["tomada"] for plano in _planos(lista)} <= existentes


def test_acoes_citadas_existem_no_diretor(lista):
    fonte = (config.HOSPITAL_DIR / "diretor.js").read_text(encoding="utf-8")
    existentes = set(re.findall(r"^  (\w+): function", fonte, re.M))
    for plano in _planos(lista):
        for acao in plano["acoes"]:
            if not acao.startswith(("construir:", "portador:")):
                assert acao in existentes, acao


def test_robos_citados_tem_sala(lista):
    salas = set(hospital.ORDEM_DA_VISITA)
    for plano in _planos(lista):
        assert plano["robo"] is None or plano["robo"] in salas


def test_nenhuma_narracao_fica_sem_valor_preenchido(lista):
    for plano in _planos(lista):
        assert not re.search(r"\{|\}|None|nan", plano["narracao"]), plano["id"]


def test_contador_final_aparece_na_narracao_quando_e_contagem(lista):
    """Quadro e narracao precisam dizer o mesmo numero."""
    plano = _por_id(lista, "fase1", "base")
    for item in plano["evidencias"]:
        assert str(item["numero"]) in plano["narracao"]


def test_regua_conta_a_troca_real_entre_os_erros():
    vitrine = _vitrine()
    pagina = config.VITRINE_PAGINA
    if not pagina.exists():
        pytest.skip("sem a vitrine montada")
    vitrine = hospital.casos_da_vitrine(pagina)
    lista = hospital.montar_dados(vitrine)["filmes"]
    regua = _por_id(lista, "fase2", "regua")["narracao"]
    comparacao = json.loads((config.GENETIC_DIR / "comparacao_final.json").read_text(encoding="utf-8"))
    servido = next(b for b in comparacao if b["braco"] == filmes.BRACO_COMPLETO)
    negativos, positivos = filmes.erros_no_corte(vitrine["casos"], filmes.LIMIAR_PADRAO)
    assert f"{negativos} falsos negativos e {positivos} falsos positivos" in regua
    assert f"{servido['falsos_negativos']} e {servido['falsos_positivos']}" in regua


def test_regua_avisa_que_o_teste_so_ilustra(lista):
    assert "depois da avaliação final" in _por_id(lista, "fase2", "regua")["narracao"]


def test_filme_da_evolucao_nao_vende_modelo_melhor(lista):
    assert "não achou um modelo melhor" in _por_id(lista, "fase2", "achado")["narracao"]


def test_filme_da_fase3_se_declara_rascunho_e_guarda_lugar_para_o_real(lista):
    filme = next(f for f in lista if f["id"] == "fase3")
    assert filme["rascunho"] is True
    assert any(p["telao"] for p in filme["planos"])
    assert "ainda não funciona" in filme["planos"][0]["narracao"]


def test_figuras_dos_filmes_entram_na_pagina(lista):
    dados = hospital.montar_dados(_vitrine())
    usadas = set(hospital.figuras_usadas(dados))
    assert {p["figura"] for p in _planos(lista) if p["figura"]} <= usadas

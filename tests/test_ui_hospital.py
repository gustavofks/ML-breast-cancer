"""Hospital 3D: as falas sao fieis aos artefatos, e a pagina e autossuficiente."""

from __future__ import annotations

import json
import re

import pytest

from src import config
from src.ui import hospital


def _casos_minimos() -> dict:
    caso = {
        "diagnostico_registrado": "Maligno",
        "classe": "Benigno",
        "probabilidade": 0.1,
        "limiar": 0.2,
        "medidas": {"texture_worst": 1.0},
        "contribuicoes": [{"feature": "texture_worst", "contribuicao": -1.0}] * 5,
        "texto": "## Leitura\nTexto.",
        "texto_motivo": None,
    }
    evento = {"evento": "requisicao", "metodo": "GET", "rota": "/health", "status": 200, "latencia_ms": 1.0}
    return {"casos": [caso], "console": {"gravacao": {"eventos": [evento]}}}


@pytest.fixture(scope="module")
def dados() -> dict:
    return hospital.montar_dados(_casos_minimos())


def _falas(sala: dict) -> list[str]:
    return [sala["fala"], *(p["resposta"] for p in sala["perguntas"])]


def test_visita_passa_por_todas_as_salas_na_ordem_da_historia(dados):
    assert tuple(s["id"] for s in dados["salas"]) == hospital.ORDEM_DA_VISITA
    assert set(hospital.ORDEM_DA_VISITA) == set(hospital.CONSTRUTORES)


def test_toda_resposta_traz_a_fonte(dados):
    for sala in dados["salas"]:
        for pergunta in sala["perguntas"]:
            assert pergunta["fonte"], pergunta["pergunta"]


def test_fontes_que_sao_arquivos_do_repositorio_existem(dados):
    """Documentos ficam de fora: nem todos existem no container."""
    for sala in dados["salas"]:
        for pergunta in sala["perguntas"]:
            fonte = pergunta["fonte"]
            if fonte.startswith(("src/", "results/")):
                assert (config.PROJECT_ROOT / fonte).exists(), fonte


def test_nenhuma_fala_fica_sem_valor_preenchido(dados):
    for sala in dados["salas"]:
        for fala in _falas(sala):
            assert not re.search(r"\{|\}|None|nan", fala), fala


def test_fala_dos_dados_repete_as_contagens_do_dataset(dados):
    base = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))["dataset"]
    fala = next(s for s in dados["salas"] if s["id"] == "dados")["fala"]
    for chave in ("benignos", "malignos", "treino", "teste"):
        assert str(base[chave]) in fala


def test_robo_da_evolucao_nao_vende_modelo_melhor(dados):
    """O achado central da Fase 2: o ganho vem do ponto de corte."""
    sala = next(s for s in dados["salas"] if s["id"] == "evolucao")
    resposta = sala["perguntas"][0]
    assert resposta["pergunta"] == "O algoritmo achou um modelo melhor?"
    assert resposta["resposta"].startswith("Não")
    assert "ponto de corte" in resposta["resposta"]


def test_enxame_usa_a_execucao_do_modelo_servido(dados):
    execucoes = json.loads((config.GENETIC_DIR / "execucoes.json").read_text(encoding="utf-8"))
    servida = hospital.execucao_servida(execucoes)
    origem = json.loads(config.MODELO_SERVIDO_FILE.read_text(encoding="utf-8"))["origem"]
    assert servida["experimento"] in origem
    assert f"semente {servida['semente']}" in origem
    enxame = next(s for s in dados["salas"] if s["id"] == "evolucao")["enxame"]
    assert len(enxame["historico"]) == len(servida["historico"])
    assert enxame["populacao"] == servida["configuracao"]["populacao"]


def test_ala_em_obras_avisa_que_ainda_nao_funciona(dados):
    obras = next(s for s in dados["salas"] if s["id"] == "obras")
    assert "nada funciona ainda" in obras["fala"]
    assert "Validação humana" in obras["fluxo"]


def test_aviso_igual_ao_da_interface_do_servico():
    """Copia necessaria: `src/ui/app.py` traz o pandas, que este modulo evita."""
    pytest.importorskip("pandas")
    from src.ui.app import AVISO

    assert AVISO.replace("**", "") == hospital.AVISO


def test_caso_resumido_leva_so_as_tres_maiores_contribuicoes(dados):
    assert len(dados["casos"][0]["contribuicoes"]) == 3


def test_figuras_citadas_existem_e_embutem_como_png(dados):
    figuras = hospital.embutir_figuras(dados)
    assert set(figuras) == set(hospital.figuras_usadas(dados))
    assert all(uri.startswith("data:image/png;base64,") for uri in figuras.values())


def test_pagina_nao_carrega_nada_de_fora_e_nao_sobra_marcador(dados):
    html = hospital.pagina(dados, {})
    sem_three = re.sub(r'<script id="three-fonte"[\s\S]*?</script>', "", html)
    assert not re.search(r'(?:src|href)="https?://', sem_three)
    assert not re.search(r"__[A-Z]+__", sem_three)


def test_dados_nao_fecham_o_bloco_de_script():
    assert "</script" not in hospital._json_seguro({"texto": "</script><b>"})


def test_vitrine_ausente_explica_como_montar(tmp_path):
    with pytest.raises(FileNotFoundError, match="montar_space"):
        hospital.casos_da_vitrine(tmp_path / "nada.html")

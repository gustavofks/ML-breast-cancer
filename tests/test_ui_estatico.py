"""Testes da versao estatica do painel, publicada como Space estatico."""

import json
import re

import pytest

pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main, servico  # noqa: E402
from src.ui import app as painel  # noqa: E402
from src.ui import estatico  # noqa: E402


@pytest.fixture(scope="module")
def api():
    """A API de verdade, com o modelo servido e o cache de textos do projeto."""
    return TestClient(main.criar_app(servico.Servico.construir(servico.carregar_especificacao())))


@pytest.fixture(scope="module")
def dados(api):
    return estatico.dados_dos_casos(api)


def test_um_registro_por_caso_do_conjunto_de_teste(dados):
    assert len(dados["casos"]) == len(painel.casos_de_teste())


def test_numeros_vem_da_propria_api(api, dados):
    """A pagina estatica mostra o que a API serve, nao uma conta paralela."""
    medidas = painel.casos_de_teste().drop(columns="diagnostico_registrado").iloc[3].to_dict()
    resposta = api.post("/explain", json={"medidas": medidas}).json()

    caso = dados["casos"][3]
    assert caso["probabilidade"] == resposta["probabilidade"]
    assert caso["classe"] == resposta["classe"]
    assert caso["contribuicoes"] == resposta["contribuicoes"]
    assert caso["texto"] == resposta["texto"]


def test_registro_traz_o_diagnostico_registrado_e_o_modelo_servido(dados):
    assert dados["casos"][0]["diagnostico_registrado"] in {"Benigno", "Maligno"}
    assert dados["modelo"]["origem"]
    assert dados["modelo"]["limiar"]


def test_pagina_traz_o_aviso_permanente(dados):
    assert "diagnóstico final é sempre do médico" in estatico.pagina(dados)


def test_dados_embutidos_nao_fecham_o_script_antes_da_hora(dados):
    """Texto com "</script>" dentro do JSON encerraria o bloco e quebraria a pagina."""
    dados = json.loads(json.dumps(dados))
    dados["casos"][0]["texto"] = "ataque </script><script>alert(1)</script>"

    html = estatico.pagina(dados)
    bloco = re.search(r'<script id="dados" type="application/json">(.*?)</script>', html, re.S)

    assert json.loads(bloco.group(1))["casos"][0]["texto"].startswith("ataque </script>")


def test_pagina_nao_depende_de_nada_externo(dados):
    """Space estatico abre sem servidor; recurso externo e ponto de falha a mais."""
    html = estatico.pagina(dados)

    assert not re.search(r'(src|href)="https?://', html)

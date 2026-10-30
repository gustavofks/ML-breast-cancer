"""Testes da vitrine: dados vindos dos artefatos e da API, e a pagina montada."""

import json
import re
import shutil
import subprocess

import pytest

pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")

from src import config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.preprocessing import split_data  # noqa: E402
from src.ui import vitrine  # noqa: E402


# ---------------------------------------------------------------------------
# Processo e textos, a partir dos artefatos
# ---------------------------------------------------------------------------
def test_convergencia_e_a_media_das_sementes_so_do_braco_completo(payload_ag):
    execucoes = [
        payload_ag(semente=42, fitness=0.90),
        payload_ag(semente=43, fitness=0.94),
        payload_ag(semente=44, fitness=0.50, braco="AG, só limiar"),
    ]

    curvas = vitrine.convergencia(execucoes)

    assert list(curvas) == ["E1 base"]
    assert curvas["E1 base"]["Regressão Logística"] == pytest.approx([0.91, 0.92])


def test_processo_sem_artefatos_devolve_none_e_a_tela_diz_que_falta(tmp_path):
    assert vitrine.processo(tmp_path) is None
    assert vitrine.textos(tmp_path) is None


def _narrativa(identificador, aprovado=True, falha=None):
    verificacoes = {
        nome: {"aprovado": nome != falha, "detalhes": []}
        for nome in ("fidelidade", "seguranca", "cobertura", "formato")
    }
    return {
        "identificador": identificador,
        "texto": f"## Titulo\ntexto de {identificador}",
        "qualidade": {"aprovado": aprovado, "verificacoes": verificacoes},
    }


def test_textos_contam_aprovacao_por_verificacao_e_separam_os_de_gestao(tmp_path):
    llm = tmp_path / "llm"
    llm.mkdir()
    narrativas = [
        _narrativa("explicacao_paciente_v2"),
        _narrativa("explicacao_paciente_v2", aprovado=False, falha="seguranca"),
        _narrativa("resumo_gestao_v1"),
    ]
    (llm / "narrativas.json").write_text(json.dumps(narrativas), encoding="utf-8")

    textos = vitrine.textos(tmp_path)

    assert textos["total"] == 3
    assert textos["aprovados"] == 2
    assert textos["verificacoes"]["seguranca"] == 2
    assert textos["verificacoes"]["formato"] == 3
    assert [g["identificador"] for g in textos["gestao"]] == ["resumo_gestao_v1"]


def test_processo_real_traz_o_que_os_graficos_usam():
    processo = vitrine.processo(config.RESULTS_DIR)

    assert processo["pesos"]["falso_negativo"] > processo["pesos"]["falso_positivo"]
    assert {linha["braco"] for linha in processo["teste"]} >= {"Original", "AG, completo"}
    for experimento, curvas in processo["convergencia"].items():
        geracoes = processo["configuracoes"][experimento]["geracoes"]
        assert all(len(curva) == geracoes for curva in curvas.values())


# ---------------------------------------------------------------------------
# Faixas e cenarios: so o treino
# ---------------------------------------------------------------------------
FEATURES = ["texture_mean", "concave points_mean", "area_se"]


def test_faixas_e_perfis_vem_do_treino_e_nao_do_teste():
    X, y = load_dataset()
    X_treino, _, y_treino, _ = split_data(X, y)

    faixas = vitrine.faixas(FEATURES)

    for nome in FEATURES:
        assert faixas[nome]["minimo"] == X_treino[nome].min()
        assert faixas[nome]["maximo"] == X_treino[nome].max()
        assert faixas[nome]["media_maligno"] == pytest.approx(X_treino[nome][y_treino == 1].mean())


def test_cenario_de_desvio_passa_do_limiar_da_sentinela():
    faixas = vitrine.faixas(FEATURES)
    cenarios = {c["id"]: c for c in vitrine.cenarios(faixas)}

    medidas = cenarios["predict-desvio"]["corpo"]["medidas"]
    desvios = [abs(medidas[n] - f["media"]) / f["desvio"] for n, f in faixas.items()]

    assert max(desvios) > config.API_LIMIAR_DE_DESVIO


def test_cenario_de_medida_faltante_tira_exatamente_uma_medida():
    cenarios = {c["id"]: c for c in vitrine.cenarios(vitrine.faixas(FEATURES))}

    assert len(cenarios["predict-faltante"]["corpo"]["medidas"]) == len(FEATURES) - 1


def test_toda_rota_publica_tem_um_cenario_no_console():
    rotas = {c["rota"] for c in vitrine.cenarios(vitrine.faixas(FEATURES))}

    assert rotas == {"/health", "/model-info", "/predict", "/explain", "/metrics"}


# ---------------------------------------------------------------------------
# Montagem completa, com a API em processo
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dados():
    from src.api import servico

    return vitrine.montar_dados(servico.Servico.construir(servico.carregar_especificacao()))


def test_casos_trazem_as_medidas_que_o_simulador_usa(dados):
    assert len(dados["casos"]) == 114
    for caso in dados["casos"]:
        assert list(caso["medidas"]) == dados["modelo"]["features"]


def test_gravacao_do_console_traz_respostas_reais_de_cada_cenario(dados):
    respostas = dados["console"]["gravacao"]["respostas"]

    assert respostas["health"]["status"] == 200
    assert respostas["predict-faltante"]["status"] == 422
    assert respostas["predict-desvio"]["corpo"]["alertas"]
    assert "api_alertas_de_desvio_total" in dados["console"]["gravacao"]["metricas"]


def test_gravacao_nao_carrega_as_chamadas_da_montagem_dos_casos(dados):
    """Os 114 /explain da montagem nao podem aparecer no log gravado do console."""
    eventos = dados["console"]["gravacao"]["eventos"]
    requisicoes = [e for e in eventos if e["evento"] == "requisicao"]

    assert len(requisicoes) == len(dados["console"]["cenarios"])
    assert all(e["id"].startswith("gravacao-") for e in eventos)


def test_pagina_traz_o_aviso_permanente(dados):
    assert "diagnóstico final é sempre do médico" in vitrine.pagina(dados)


def test_pagina_nao_carrega_nada_de_fora(dados):
    """Space estatico abre sem servidor; recurso externo e ponto de falha a mais.

    Link para o repositorio e permitido: e navegacao, nao recurso carregado. O
    icone vazio em `data:` tambem: evita que o navegador peca `/favicon.ico` e
    suje as metricas com um 404.
    """
    html = vitrine.pagina(dados)

    assert not re.search(r"<script[^>]+src=", html)
    assert not re.search(r'<link\b(?![^>]*href="data:)', html)
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?https?:", html)


def test_dados_embutidos_nao_fecham_o_script_antes_da_hora(dados):
    """Texto com "</script>" dentro do JSON encerraria o bloco e quebraria a pagina."""
    dados = json.loads(json.dumps(dados))
    dados["casos"][0]["texto"] = "ataque </script><script>alert(1)</script>"

    html = vitrine.pagina(dados)
    bloco = re.search(r'<script id="dados" type="application/json">(.*?)</script>', html, re.S)

    assert json.loads(bloco.group(1))["casos"][0]["texto"].startswith("ataque </script>")


def test_nenhum_script_embutido_fecha_o_proprio_bloco():
    """O Chart.js e os scripts entram crus num <script>; um "</script" no meio quebraria tudo."""
    for nome in ("vendor/chart.umd.min.js", "logica.js", "vitrine.js"):
        assert "</script" not in (config.VITRINE_DIR / nome).read_text(encoding="utf-8").lower()


def test_script_da_tela_nunca_insere_html():
    """Texto gerado pelo modelo de linguagem e resposta da API so entram por textContent."""
    for nome in ("logica.js", "vitrine.js"):
        codigo = (config.VITRINE_DIR / nome).read_text(encoding="utf-8")
        for proibido in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert proibido not in codigo, f"{proibido} em {nome}"


def test_chartjs_versionado_traz_versao_e_licenca():
    cabeca = (config.VITRINE_DIR / "vendor" / "chart.umd.min.js").read_text(encoding="utf-8")[:200]

    assert "Chart.js v4.5.1" in cabeca
    assert (config.VITRINE_DIR / "vendor" / "chart.js-LICENSE.md").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="testes do JavaScript precisam do Node")
def test_testes_do_javascript_passam():
    resultado = subprocess.run(
        ["node", "--test", "tests/js/*.test.js"],
        capture_output=True,
        text=True,
        cwd=config.PROJECT_ROOT,
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr

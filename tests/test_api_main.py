"""Testes dos contratos das rotas."""

import pytest

pytest.importorskip("fastapi", reason="Fase 2: requer requirements-api.txt")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import main, servico  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.preprocessing import split_data  # noqa: E402


@pytest.fixture(scope="module")
def modelo():
    return servico.Servico.construir(servico.referencia_da_fase_1())


@pytest.fixture
def cliente(modelo, tmp_path):
    return TestClient(main.criar_app(modelo, diretorio_cache=tmp_path / "cache"))


@pytest.fixture(scope="module")
def caso():
    X, y = load_dataset()
    _, X_test, _, _ = split_data(X, y)
    return X_test.iloc[0].to_dict()


def test_health_confirma_que_o_modelo_carregou(cliente):
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok", "modelo_carregado": True}


def test_model_info_declara_origem_features_limiar_e_commit(cliente, monkeypatch):
    monkeypatch.setenv("APP_COMMIT", "abc1234")

    corpo = cliente.get("/model-info").json()

    assert corpo["familia"] == "Regressão Logística"
    assert len(corpo["features"]) == 30
    assert corpo["limiar"] == 0.5
    assert "Fase 1" in corpo["origem"]
    assert corpo["commit"] == "abc1234"


def test_model_info_sem_commit_declara_que_nao_sabe(cliente, monkeypatch):
    monkeypatch.delenv("APP_COMMIT", raising=False)

    assert cliente.get("/model-info").json()["commit"] == "desconhecido"


def test_predict_devolve_probabilidade_classe_e_limiar(cliente, caso, modelo):
    resposta = cliente.post("/predict", json={"medidas": caso})

    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["probabilidade"] == pytest.approx(modelo.prever(caso).probabilidade)
    assert corpo["limiar"] == 0.5
    assert corpo["classe"] in {"Benigno", "Maligno"}
    assert corpo["alertas"] == []


def test_predict_com_medida_faltante_responde_422_nomeando_a_medida(cliente, caso):
    incompleto = {k: v for k, v in caso.items() if k != "area_worst"}

    resposta = cliente.post("/predict", json={"medidas": incompleto})

    assert resposta.status_code == 422
    assert "area_worst" in resposta.text


def test_predict_com_medida_fora_da_distribuicao_alerta_e_ainda_responde(cliente, caso):
    """A sentinela registra; quem decide se confia no resultado e o medico."""
    deslocado = {**caso, "area_worst": 1e6}

    corpo = cliente.post("/predict", json={"medidas": deslocado}).json()

    assert [a["feature"] for a in corpo["alertas"]] == ["area_worst"]
    assert "probabilidade" in corpo


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from src.api import explicacao  # noqa: E402
from src.llm import narrator, prompts  # noqa: E402
from src.llm.provider import CachedProvider, FakeProvider  # noqa: E402

TEXTO_APROVAVEL = """## Leitura do caso
Texto de teste.

## Medidas que mais pesaram
Pesaram {citacao}.

## Ressalva
Esta leitura é apoio à decisão e não substitui a avaliação do médico responsável.
"""


def _semear_cache(modelo, caso, diretorio, texto):
    """Grava no cache o que `run_llm.py` gravaria para este caso."""
    pares = explicacao.contribuicoes(modelo, pd.DataFrame([caso]))
    previsao = modelo.prever(caso)
    contexto = narrator.contexto_do_paciente(
        probabilidade=previsao.probabilidade, limiar=previsao.limiar, contribuicoes=pares
    )
    citacao = ", ".join(c["feature"] for c in contexto["maiores_contribuicoes"])
    narrator.narrar(
        prompts.TEMPLATES["explicacao_paciente_v2"],
        contexto,
        CachedProvider(FakeProvider([texto.format(citacao=citacao)]), diretorio),
    )


def test_explain_traz_contribuicoes_ordenadas_por_peso(cliente, caso):
    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    pesos = [abs(c["contribuicao"]) for c in corpo["contribuicoes"]]
    assert pesos == sorted(pesos, reverse=True)
    assert len(corpo["contribuicoes"]) == 30
    assert {"probabilidade", "limiar", "classe", "alertas"} <= set(corpo)


def test_explain_sem_texto_no_cache_diz_por_que(cliente, caso):
    """Geracao e offline; caso que run_llm.py nao processou nao tem texto."""
    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    assert corpo["texto"] is None
    assert "run_llm.py" in corpo["texto_motivo"]


def test_explain_devolve_o_texto_gerado_offline(modelo, caso, tmp_path):
    """O cache gravado offline e encontrado pela mesma chave, online."""
    cache = tmp_path / "cache"
    _semear_cache(modelo, caso, cache, TEXTO_APROVAVEL)
    cliente = TestClient(main.criar_app(modelo, diretorio_cache=cache))

    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    assert corpo["texto"] is not None
    assert "não substitui" in corpo["texto"]
    assert corpo["texto_motivo"] is None


def test_texto_reprovado_nas_verificacoes_nao_chega_a_tela(modelo, caso, tmp_path):
    """Texto que falhou a verificacao de seguranca nao e mostrado a um medico."""
    cache = tmp_path / "cache"
    _semear_cache(modelo, caso, cache, "O diagnóstico é câncer confirmado.")
    cliente = TestClient(main.criar_app(modelo, diretorio_cache=cache))

    corpo = cliente.post("/explain", json={"medidas": caso}).json()

    assert corpo["texto"] is None
    assert "seguranca" in corpo["texto_motivo"]


def test_explain_com_medida_faltante_responde_422(cliente, caso):
    incompleto = {k: v for k, v in caso.items() if k != "area_worst"}

    assert cliente.post("/explain", json={"medidas": incompleto}).status_code == 422


# ---------------------------------------------------------------------------
# Log por requisicao, metricas e identificador
# ---------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

from src.api import registro  # noqa: E402


def test_metrics_conta_as_requisicoes_por_rota_e_status(cliente, caso):
    cliente.post("/predict", json={"medidas": caso})
    cliente.post("/predict", json={"medidas": {}})

    texto = cliente.get("/metrics").text

    assert 'api_requisicoes_total{rota="/predict",status="200"} 1' in texto
    assert 'api_requisicoes_total{rota="/predict",status="422"} 1' in texto
    assert 'api_latencia_segundos_count{rota="/predict"} 2' in texto


def test_metrics_responde_em_texto_puro(cliente):
    """O coletor do Prometheus le texto, nao JSON com o texto dentro de uma string."""
    resposta = cliente.get("/metrics")

    assert resposta.headers["content-type"].startswith("text/plain")
    assert resposta.text.startswith("# HELP")


def test_rota_inexistente_nao_cria_um_rotulo_por_caminho(cliente):
    """Rotulo por caminho bruto explode a cardinalidade com qualquer varredura."""
    for caminho in ("/a", "/b", "/c"):
        cliente.get(caminho)

    texto = cliente.get("/metrics").text

    assert 'rota="/a"' not in texto
    assert 'api_requisicoes_total{rota="nao_encontrada",status="404"} 3' in texto


def test_resposta_devolve_o_identificador_recebido(cliente):
    """O mesmo id do log volta no cabecalho, para cruzar reclamacao e linha de log."""
    resposta = cliente.get("/health", headers={"X-Request-ID": "pedido-1"})

    assert resposta.headers["X-Request-ID"] == "pedido-1"


def test_sem_identificador_a_api_gera_um(cliente):
    primeira = cliente.get("/health").headers["X-Request-ID"]
    segunda = cliente.get("/health").headers["X-Request-ID"]

    assert primeira and segunda and primeira != segunda


def test_alerta_de_desvio_aparece_nas_metricas(cliente, caso):
    cliente.post("/predict", json={"medidas": {**caso, "area_worst": 1e6}})

    assert 'api_alertas_de_desvio_total{feature="area_worst"} 1' in cliente.get("/metrics").text


def test_cada_requisicao_vira_uma_linha_de_log_com_o_resultado(modelo, caso, tmp_path, monkeypatch):
    """Secao 6.2: id, rota, latencia, status e resultado, uma linha por requisicao."""
    fluxo = io.StringIO()
    original = registro.configurar
    monkeypatch.setattr(registro, "configurar", lambda nome="api": original(nome, fluxo))
    cliente = TestClient(main.criar_app(modelo, diretorio_cache=tmp_path / "cache"))

    resposta = cliente.post("/predict", json={"medidas": caso}, headers={"X-Request-ID": "r1"})

    linhas = [json.loads(l) for l in fluxo.getvalue().splitlines()]
    requisicoes = [l for l in linhas if l["evento"] == "requisicao"]
    assert len(requisicoes) == 1
    linha = requisicoes[0]
    assert linha["id"] == "r1"
    assert linha["rota"] == "/predict"
    assert linha["status"] == 200
    assert linha["resultado"] == resposta.json()["classe"]
    assert linha["latencia_ms"] >= 0


def test_alerta_de_desvio_vai_ao_log_com_o_id_da_requisicao(modelo, caso, tmp_path, monkeypatch):
    fluxo = io.StringIO()
    original = registro.configurar
    monkeypatch.setattr(registro, "configurar", lambda nome="api": original(nome, fluxo))
    cliente = TestClient(main.criar_app(modelo, diretorio_cache=tmp_path / "cache"))

    cliente.post(
        "/predict", json={"medidas": {**caso, "area_worst": 1e6}}, headers={"X-Request-ID": "r2"}
    )

    desvios = [
        json.loads(l) for l in fluxo.getvalue().splitlines() if '"desvio"' in l
    ]
    assert [(d["id"], d["feature"]) for d in desvios] == [("r2", "area_worst")]


# ---------------------------------------------------------------------------
# Log recente, lido pela vitrine
# ---------------------------------------------------------------------------
def _cliente_com_log_proprio(modelo, tmp_path, nome):
    logger = registro.configurar(nome, io.StringIO())
    return TestClient(main.criar_app(modelo, diretorio_cache=tmp_path / "cache", logger=logger))


def test_logs_recentes_traz_requisicoes_e_desvios_com_o_id(modelo, caso, tmp_path):
    cliente = _cliente_com_log_proprio(modelo, tmp_path, "teste-logs-recentes")

    cliente.post("/predict", json={"medidas": {**caso, "area_worst": 1e6}},
                 headers={"X-Request-ID": "r3"})
    eventos = cliente.get("/logs/recentes").json()["eventos"]

    assert {(e["evento"], e["id"]) for e in eventos} == {("requisicao", "r3"), ("desvio", "r3")}


def test_logs_recentes_devolve_so_o_que_veio_depois(modelo, tmp_path):
    cliente = _cliente_com_log_proprio(modelo, tmp_path, "teste-logs-depois")
    cliente.get("/health")
    ultimo = cliente.get("/logs/recentes").json()["eventos"][-1]["seq"]

    cliente.get("/health", headers={"X-Request-ID": "novo"})
    eventos = cliente.get("/logs/recentes", params={"depois": ultimo}).json()["eventos"]

    assert [e["id"] for e in eventos] == ["novo"]


def test_ler_o_log_nao_gera_log_nem_metrica(modelo, tmp_path):
    """Cada leitura geraria uma linha nova, e a tela leria o proprio eco para sempre."""
    cliente = _cliente_com_log_proprio(modelo, tmp_path, "teste-logs-silencio")

    for _ in range(3):
        cliente.get("/logs/recentes")

    assert cliente.get("/logs/recentes").json()["eventos"] == []
    assert "/logs/recentes" not in cliente.get("/metrics").text


def test_duas_apis_com_loggers_proprios_nao_misturam_o_log(modelo, tmp_path):
    """A vitrine monta seus dados numa segunda API; o log dela nao vaza para a servida."""
    servida = _cliente_com_log_proprio(modelo, tmp_path, "teste-servida")
    montagem = _cliente_com_log_proprio(modelo, tmp_path, "teste-montagem")

    montagem.get("/health")

    assert servida.get("/logs/recentes").json()["eventos"] == []

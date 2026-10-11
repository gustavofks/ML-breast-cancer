"""Testes da sentinela de desvio e dos contadores."""

import pytest

from src import config
from src.api import monitoring
from src.data import load_dataset
from src.preprocessing import split_data


@pytest.fixture(scope="module")
def treino():
    X, y = load_dataset()
    X_train, _, _, _ = split_data(X, y)
    return X_train


def test_sentinela_quase_nao_alerta_sobre_o_proprio_treino(treino):
    """Alerta que dispara em caso legitimo vira ruido, e ruido e ignorado.

    Com k = 3, 13% do treino alertaria. O limiar configurado tem de manter a
    taxa abaixo de 5% nos dados que definem o normal.
    """
    sentinela = monitoring.Sentinela.de_treino(treino, k=config.API_LIMIAR_DE_DESVIO)

    disparos = sum(bool(sentinela.verificar(linha.to_dict())) for _, linha in treino.iterrows())

    assert disparos / len(treino) < 0.05


def test_medida_deslocada_dispara_alerta_nomeado(treino):
    """O caso da Fase 1: equipamento recalibrado desloca uma medida sem erro visivel."""
    sentinela = monitoring.Sentinela.de_treino(treino, k=5.0)
    caso = treino.iloc[0].to_dict()
    caso["area_worst"] = treino["area_worst"].mean() + 50 * treino["area_worst"].std(ddof=0)

    alertas = sentinela.verificar(caso)

    assert [a.feature for a in alertas] == ["area_worst"]
    assert alertas[0].desvios > 5.0


def test_desvio_para_baixo_tambem_alerta(treino):
    """Medida zerada por falha de sensor e tao suspeita quanto medida enorme."""
    sentinela = monitoring.Sentinela.de_treino(treino, k=5.0)
    caso = treino.iloc[0].to_dict()
    caso["radius_mean"] = treino["radius_mean"].mean() - 50 * treino["radius_mean"].std(ddof=0)

    assert [a.feature for a in sentinela.verificar(caso)] == ["radius_mean"]


def test_medida_que_a_sentinela_nao_conhece_e_ignorada(treino):
    """Quem recusa medida desconhecida e a validacao do servico, nao a sentinela."""
    sentinela = monitoring.Sentinela.de_treino(treino, k=5.0)

    assert sentinela.verificar({"outra_coisa": 1e9}) == []


def test_metricas_saem_no_formato_texto_do_prometheus():
    monitor = monitoring.Monitor()
    monitor.registrar("/predict", 200, 0.010)
    monitor.registrar("/predict", 200, 0.030)
    monitor.registrar("/predict", 422, 0.001)

    texto = monitor.exportar()

    assert "# TYPE api_requisicoes_total counter" in texto
    assert 'api_requisicoes_total{rota="/predict",status="200"} 2' in texto
    assert 'api_requisicoes_total{rota="/predict",status="422"} 1' in texto
    assert 'api_latencia_segundos_count{rota="/predict"} 3' in texto
    assert 'api_latencia_segundos_sum{rota="/predict"} 0.041' in texto


def test_toda_linha_de_amostra_tem_metrica_declarada():
    """Amostra sem `# TYPE` correspondente e rejeitada pelo coletor do Prometheus."""
    monitor = monitoring.Monitor()
    monitor.registrar("/health", 200, 0.001)
    monitor.registrar_alertas([monitoring.Alerta("area_worst", 1e6, 42.0)])

    linhas = monitor.exportar().splitlines()
    declaradas = {l.split()[2] for l in linhas if l.startswith("# TYPE")}
    amostras = [l for l in linhas if l and not l.startswith("#")]

    for amostra in amostras:
        nome = amostra.split("{")[0]
        base = nome.removesuffix("_sum").removesuffix("_count")
        assert base in declaradas, amostra


def test_alertas_viram_contador_por_medida():
    monitor = monitoring.Monitor()
    monitor.registrar_alertas([monitoring.Alerta("area_worst", 1e6, 42.0)])
    monitor.registrar_alertas([monitoring.Alerta("area_worst", 2e6, 84.0)])

    assert 'api_alertas_de_desvio_total{feature="area_worst"} 2' in monitor.exportar()

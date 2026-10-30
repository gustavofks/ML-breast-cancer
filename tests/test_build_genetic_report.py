"""Testes do gerador de artefatos do algoritmo genetico."""

import importlib.util
import json

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "build_genetic_report.py"
    spec = importlib.util.spec_from_file_location("build_genetic_report", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture
def entradas(tmp_path, payload_ag):
    """Um `execucoes.json` e um `metrics.json` minimos em disco."""
    execucoes = tmp_path / "execucoes.json"
    execucoes.write_text(
        json.dumps([payload_ag(semente=42), payload_ag(semente=43)], ensure_ascii=False),
        encoding="utf-8",
    )

    metricas = tmp_path / "metrics.json"
    metricas.write_text(
        json.dumps(
            {
                "explicabilidade": {
                    "shap_global": [
                        {"feature": "texture_worst", "shap_medio": 1.25},
                        {"feature": "area_worst", "shap_medio": 0.8},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return execucoes, metricas


def test_gera_o_resumo_e_as_tres_figuras(script, entradas, tmp_path):
    execucoes, metricas = entradas
    destino = tmp_path / "saida"

    resumo = script.construir(execucoes, metricas, destino)

    assert set(resumo) == {"execucoes", "experimentos", "bracos", "features", "varredura"}
    assert (destino / "summary.json").exists()
    for nome in script.FIGURAS.values():
        assert (destino / nome).exists()


def test_resumo_gravado_e_o_mesmo_que_foi_devolvido(script, entradas, tmp_path):
    """O relatorio le o arquivo; se ele divergir do calculo, o numero publicado erra."""
    execucoes, metricas = entradas
    destino = tmp_path / "saida"

    resumo = script.construir(execucoes, metricas, destino)
    gravado = json.loads((destino / "summary.json").read_text(encoding="utf-8"))

    assert gravado == resumo


def test_metricas_sem_secao_de_shap_nao_derrubam_a_geracao(script, tmp_path, payload_ag):
    """O cruzamento com SHAP e um extra; a ausencia dele nao pode travar o resto."""
    execucoes = tmp_path / "execucoes.json"
    execucoes.write_text(json.dumps([payload_ag()], ensure_ascii=False), encoding="utf-8")
    metricas = tmp_path / "metrics.json"
    metricas.write_text(json.dumps({}), encoding="utf-8")

    resumo = script.construir(execucoes, metricas, tmp_path / "saida")

    assert resumo["features"]
    assert all(linha["posicao_shap"] is None for linha in resumo["features"])


def test_geracao_e_idempotente(script, entradas, tmp_path):
    """Rodar de novo regenera os mesmos artefatos, como o pipeline da Fase 1."""
    execucoes, metricas = entradas
    destino = tmp_path / "saida"

    script.construir(execucoes, metricas, destino)
    primeiro = (destino / "summary.json").read_bytes()
    script.construir(execucoes, metricas, destino)

    assert (destino / "summary.json").read_bytes() == primeiro


def _varredura_em_disco(tmp_path, payload_ag):
    """Um `varredura.json` minimo, com duas politicas de custo."""
    payloads = []
    for razao in (5, 20):
        payload = payload_ag()
        payload["custo"] = {
            "falso_negativo": float(razao),
            "falso_positivo": 1.0,
            "parcimonia": 0.0,
        }
        payloads.append(payload)

    caminho = tmp_path / "varredura.json"
    caminho.write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
    return caminho


def test_varredura_entra_no_resumo_e_ganha_figura_propria(
    script, entradas, tmp_path, payload_ag
):
    """Secao 4.5.1: a curva que torna a escolha da razao de custo informada."""
    execucoes, metricas = entradas
    destino = tmp_path / "saida"

    resumo = script.construir(
        execucoes, metricas, destino, varredura=_varredura_em_disco(tmp_path, payload_ag)
    )

    assert [linha["razao_custo"] for linha in resumo["varredura"]] == [5.0, 20.0]
    assert (destino / script.FIGURA_VARREDURA).exists()


def test_sem_varredura_o_resumo_declara_a_ausencia(script, entradas, tmp_path):
    """A varredura roda separada; sem ela, o resumo diz que nao ha, e nao some.

    Lista vazia explicita, e nenhuma figura: uma figura em branco seria lida
    como resultado.
    """
    execucoes, metricas = entradas
    destino = tmp_path / "saida"

    resumo = script.construir(
        execucoes, metricas, destino, varredura=tmp_path / "nao_existe.json"
    )

    assert resumo["varredura"] == []
    assert not (destino / script.FIGURA_VARREDURA).exists()


def test_varredura_nao_se_mistura_aos_experimentos(script, entradas, tmp_path, payload_ag):
    """As execucoes da varredura nao entram na contagem nem na tabela de bracos."""
    execucoes, metricas = entradas

    sem = script.construir(execucoes, metricas, tmp_path / "a")
    com = script.construir(
        execucoes, metricas, tmp_path / "b", varredura=_varredura_em_disco(tmp_path, payload_ag)
    )

    assert com["execucoes"] == sem["execucoes"]
    assert com["bracos"] == sem["bracos"]

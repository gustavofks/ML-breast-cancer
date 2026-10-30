"""Testes da escolha do modelo servido."""

import importlib.util
import json

import pytest

from src import config
from src.api import servico


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "escolher_modelo.py"
    spec = importlib.util.spec_from_file_location("escolher_modelo", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _execucoes(payload_ag):
    return [
        payload_ag(braco="AG, completo", semente=42, fitness=0.95, features=("area_worst",)),
        payload_ag(
            braco="AG, completo",
            semente=43,
            fitness=0.97,
            features=("texture_worst", "area_worst"),
        ),
        payload_ag(braco="AG, só limiar", semente=42, fitness=0.99, segmentos=("limiar",)),
    ]


def test_escolhe_a_semente_de_maior_fitness_do_braco(script, payload_ag):
    """Fitness em validacao cruzada no treino: o teste nunca decide a escolha."""
    especificacao = script.especificacao_do_braco(_execucoes(payload_ag), "AG, completo")

    assert especificacao["features"] == ["texture_worst", "area_worst"]
    assert "semente 43" in especificacao["origem"]


def test_especificacao_gravada_e_aceita_pela_api(script, payload_ag, tmp_path):
    """O arquivo tem de carregar sem traducao, ou a API cai na referencia."""
    destino = tmp_path / "modelo_servido.json"
    execucoes = tmp_path / "execucoes.json"
    execucoes.write_text(json.dumps(_execucoes(payload_ag), ensure_ascii=False), encoding="utf-8")

    script.gravar(script.especificacao_do_braco(_execucoes(payload_ag), "AG, completo"), destino)

    carregada = servico.carregar_especificacao(destino)
    assert carregada.familia == "Regressão Logística"
    assert carregada.limiar == 0.25
    assert "validação cruzada" in carregada.origem


def test_braco_inexistente_falha_listando_os_disponiveis(script, payload_ag):
    with pytest.raises(SystemExit, match="AG, só limiar"):
        script.especificacao_do_braco(_execucoes(payload_ag), "AG, inventado")


def test_referencia_explicita_grava_o_modelo_da_fase_1(script, tmp_path):
    """Servir a referencia e uma escolha legitima, e fica declarada no arquivo."""
    destino = tmp_path / "modelo_servido.json"

    script.gravar(script.especificacao_da_referencia(), destino)

    assert servico.carregar_especificacao(destino) == servico.referencia_da_fase_1()


def test_script_nao_le_o_conjunto_de_teste(script):
    """A escolha so pode usar o que a busca viu; ler o teste aqui seria vazamento."""
    fonte = (config.PROJECT_ROOT / "scripts" / "escolher_modelo.py").read_text(encoding="utf-8")

    assert "split_data" not in fonte
    assert "load_dataset" not in fonte

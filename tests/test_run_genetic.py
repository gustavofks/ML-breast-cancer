"""Testes do script de execucao das buscas.

O script e o unico ponto da Fase 2 que separa treino e teste, e o unico que
escreve artefato em disco. As duas propriedades testadas aqui sao exatamente
essas: a lista de execucoes corresponde ao que o enunciado pede, e o artefato
gravado e reprodutivel.
"""

import importlib.util
import json

import pytest

from src import config
from src.genetic import experiments
from src.genetic.chromosome import SEGMENTOS


@pytest.fixture(scope="module")
def script():
    """Carrega `scripts/run_genetic.py` como modulo, sem executa-lo como programa."""
    caminho = config.PROJECT_ROOT / "scripts" / "run_genetic.py"
    spec = importlib.util.spec_from_file_location("run_genetic", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_modo_smoke_executa_uma_busca_minima(script):
    """O smoke existe para provar o encanamento sem esperar horas."""
    execucoes = script.montar_execucoes(smoke=True)

    assert len(execucoes) == 1
    configuracao, familia, semente, segmentos, custo = execucoes[0]
    assert configuracao.populacao <= 10
    assert configuracao.geracoes <= 5
    assert segmentos == SEGMENTOS
    assert familia in experiments.FAMILIAS
    assert custo.falso_negativo > custo.falso_positivo


def test_experimentos_usam_todos_a_mesma_politica_de_custo(script):
    """Os quatro experimentos variam so parametros do AG; o fitness fica fixo.

    E isso que torna a comparacao entre eles interpretavel. Se a politica de
    custo variasse junto, nao daria para atribuir a diferenca a configuracao.
    """
    execucoes = script.montar_execucoes(smoke=False)

    politicas = {(c.falso_negativo, c.falso_positivo, c.parcimonia) for *_, c in execucoes}

    assert len(politicas) == 1


def test_modo_completo_cobre_experimentos_e_bracos(script):
    """Quatro configuracoes sobre tres familias com tres sementes, mais as ablacoes."""
    execucoes = script.montar_execucoes(smoke=False)

    esperado_experimentos = (
        len(experiments.EXPERIMENTOS) * len(experiments.FAMILIAS) * len(experiments.SEMENTES)
    )
    # "AG, completo" nao repete: ele ja e E1 com os tres segmentos ativos.
    esperado_bracos = (len(experiments.BRACOS) - 1) * len(experiments.SEMENTES)
    assert len(execucoes) == esperado_experimentos + esperado_bracos

    familias_cobertas = {familia for _, familia, _, _, _ in execucoes}
    assert familias_cobertas == set(experiments.FAMILIAS)

    segmentos_cobertos = {segmentos for _, _, _, segmentos, _ in execucoes}
    assert SEGMENTOS in segmentos_cobertos
    for segmentos in experiments.BRACOS.values():
        assert segmentos in segmentos_cobertos


def test_varredura_cobre_as_nove_politicas_de_custo(script):
    """Nove combinacoes de razao e alfa, da secao 4.5.1 do design."""
    execucoes = script.montar_varredura()

    assert len(execucoes) == len(experiments.VARREDURA_CUSTO) * len(experiments.SEMENTES)
    politicas = {(c.falso_negativo, c.parcimonia) for *_, c in execucoes}
    assert len(politicas) == 9


def test_varredura_mantem_a_configuracao_do_algoritmo_fixa(script):
    """A varredura mede politica clinica, nao parametro de busca."""
    execucoes = script.montar_varredura()

    configuracoes = {c.nome for c, *_ in execucoes}
    familias = {f for _, f, *_ in execucoes}

    assert len(configuracoes) == 1
    assert familias == {experiments.FAMILIA_DOS_BRACOS}


def test_payload_registra_a_politica_de_custo_usada(script, tmp_path, monkeypatch):
    """Sem isso, as nove varreduras viram nove numeros sem rotulo."""
    destino = tmp_path / "sweep.json"
    monkeypatch.setattr(
        "sys.argv", ["run_genetic.py", "--smoke", "--sweep", "--saida", str(destino)]
    )

    script.main()

    payloads = json.loads(destino.read_text(encoding="utf-8"))
    assert payloads
    for registro in payloads:
        assert registro["custo"]["falso_negativo"] > 0
        assert "parcimonia" in registro["custo"]


def test_payload_registra_a_estabilidade_dos_cinco_melhores(script, tmp_path, monkeypatch):
    """Secao 4.7: a queda sob outra particao e reportada, nao escondida."""
    destino = tmp_path / "exec.json"
    monkeypatch.setattr("sys.argv", ["run_genetic.py", "--smoke", "--saida", str(destino)])

    script.main()

    registro = json.loads(destino.read_text(encoding="utf-8"))[0]
    assert len(registro["estabilidade"]) == len(registro["melhores"])
    for item in registro["estabilidade"]:
        assert "queda" in item


def test_payload_nomeia_o_braco_pelo_conjunto_de_segmentos(script, tmp_path, monkeypatch):
    """A tabela comparativa agrupa por braco; o nome tem que vir do artefato."""
    destino = tmp_path / "exec.json"
    monkeypatch.setattr("sys.argv", ["run_genetic.py", "--smoke", "--saida", str(destino)])

    script.main()

    registro = json.loads(destino.read_text(encoding="utf-8"))[0]
    assert registro["braco"] == "AG, completo"


def test_artefato_gravado_e_reprodutivel(script, tmp_path, monkeypatch):
    """Rodar duas vezes com a mesma semente produz o mesmo arquivo, byte a byte.

    A regra vale para `results/metrics.json` desde a Fase 1 e vale aqui pelo
    mesmo motivo: artefato que muda entre execucoes nao serve de evidencia no
    relatorio. Cronometro de parede no payload quebra essa garantia sem gerar
    erro nenhum — defeito silencioso, do tipo que esta suite existe para pegar.
    """
    primeiro, segundo = tmp_path / "a.json", tmp_path / "b.json"

    for destino in (primeiro, segundo):
        monkeypatch.setattr(
            "sys.argv", ["run_genetic.py", "--smoke", "--saida", str(destino)]
        )
        script.main()

    assert primeiro.read_bytes() == segundo.read_bytes()


def test_artefato_registra_historico_e_melhor_individuo(script, tmp_path, monkeypatch):
    destino = tmp_path / "execucoes.json"
    monkeypatch.setattr("sys.argv", ["run_genetic.py", "--smoke", "--saida", str(destino)])

    script.main()

    payloads = json.loads(destino.read_text(encoding="utf-8"))
    assert len(payloads) == 1
    registro = payloads[0]
    assert registro["historico"]
    assert registro["melhor"]["n_features"] >= 1
    assert registro["avaliacao"]["n_amostras"] > 0

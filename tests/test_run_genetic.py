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


def test_varredura_nao_sobrescreve_o_artefato_dos_experimentos(script):
    """As duas buscas gravam em arquivos distintos por padrao.

    Com o mesmo destino, rodar a varredura depois dos experimentos apagaria horas
    de execucao sem erro nenhum: o arquivo continuaria valido, so que com outras
    execucoes dentro, e o relatorio seria gerado em cima delas.
    """
    experimentos = script.destino_padrao(sweep=False)
    varredura = script.destino_padrao(sweep=True)

    assert experimentos != varredura
    assert experimentos.parent == varredura.parent == config.GENETIC_DIR


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


def test_progresso_sobrevive_a_falha_no_meio_da_execucao(script, tmp_path, monkeypatch):
    """A execucao completa leva horas: queda na ultima nao pode perder as outras.

    Gravar so no fim transforma qualquer interrupcao em perda total de uma noite
    de maquina. O arquivo e regravado a cada execucao concluida.
    """
    destino = tmp_path / "parcial.json"
    evoluir_original = script.engine.evoluir
    duas_execucoes = script.montar_execucoes(smoke=True) * 2
    chamadas = {"n": 0}

    def evoluir_com_falha(*args, **kwargs):
        chamadas["n"] += 1
        if chamadas["n"] > 1:
            raise RuntimeError("falha simulada na segunda execucao")
        return evoluir_original(*args, **kwargs)

    monkeypatch.setattr(script.engine, "evoluir", evoluir_com_falha)
    monkeypatch.setattr(script, "montar_execucoes", lambda smoke: duas_execucoes)
    monkeypatch.setattr("sys.argv", ["run_genetic.py", "--smoke", "--saida", str(destino)])

    with pytest.raises(RuntimeError, match="falha simulada"):
        script.main()

    assert destino.exists()
    assert len(json.loads(destino.read_text(encoding="utf-8"))) == 1


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


def _duas_execucoes_distintas(script):
    """Duas execucoes do smoke que diferem so pela semente."""
    configuracao, familia, semente, segmentos, custo = script.montar_execucoes(smoke=True)[0]
    return [
        (configuracao, familia, semente, segmentos, custo),
        (configuracao, familia, semente + 1, segmentos, custo),
    ]


def _rodar(script, monkeypatch, execucoes, destino, *extras):
    monkeypatch.setattr(script, "montar_execucoes", lambda smoke: execucoes)
    monkeypatch.setattr(
        "sys.argv", ["run_genetic.py", "--smoke", "--saida", str(destino), *extras]
    )
    script.main()


def test_retomada_produz_o_mesmo_artefato_que_a_execucao_sem_interrupcao(
    script, tmp_path, monkeypatch
):
    """Cada execucao nasce da propria semente, entao retomar nao muda resultado.

    E a propriedade que torna a retomada segura: o arquivo final e identico, byte
    a byte, ao de uma execucao que nunca foi interrompida.
    """
    execucoes = _duas_execucoes_distintas(script)
    continua, retomada = tmp_path / "continua.json", tmp_path / "retomada.json"

    _rodar(script, monkeypatch, execucoes, continua)
    primeira = json.loads(continua.read_text(encoding="utf-8"))[:1]
    retomada.write_text(json.dumps(primeira, indent=2, ensure_ascii=False), encoding="utf-8")
    _rodar(script, monkeypatch, execucoes, retomada, "--retomar")

    assert retomada.read_bytes() == continua.read_bytes()


def test_retomada_nao_reexecuta_o_que_ja_esta_no_arquivo(script, tmp_path, monkeypatch):
    """Execucao do Random Forest custa ate meia hora; refazer e desperdicio."""
    execucoes = _duas_execucoes_distintas(script)
    destino = tmp_path / "exec.json"
    _rodar(script, monkeypatch, execucoes[:1], destino)

    evoluir_original = script.engine.evoluir
    chamadas = {"n": 0}

    def evoluir_contado(*args, **kwargs):
        chamadas["n"] += 1
        return evoluir_original(*args, **kwargs)

    monkeypatch.setattr(script.engine, "evoluir", evoluir_contado)
    _rodar(script, monkeypatch, execucoes, destino, "--retomar")

    assert chamadas["n"] == 1
    assert len(json.loads(destino.read_text(encoding="utf-8"))) == 2


def test_resultado_de_outra_configuracao_nao_e_reaproveitado(script, tmp_path, monkeypatch):
    """Mesma semente e mesmo nome de experimento nao bastam para reaproveitar.

    Se a populacao mudou entre as duas rodadas, o resultado gravado responde a
    outra pergunta, e reaproveita-lo misturaria configuracoes sem erro nenhum.
    """
    execucoes = _duas_execucoes_distintas(script)
    destino = tmp_path / "exec.json"
    _rodar(script, monkeypatch, execucoes[:1], destino)

    gravado = json.loads(destino.read_text(encoding="utf-8"))
    gravado[0]["configuracao"]["populacao"] += 1
    destino.write_text(json.dumps(gravado, ensure_ascii=False), encoding="utf-8")

    evoluir_original = script.engine.evoluir
    chamadas = {"n": 0}

    def evoluir_contado(*args, **kwargs):
        chamadas["n"] += 1
        return evoluir_original(*args, **kwargs)

    monkeypatch.setattr(script.engine, "evoluir", evoluir_contado)
    _rodar(script, monkeypatch, execucoes, destino, "--retomar")

    assert chamadas["n"] == 2


def test_sem_retomar_recusa_sobrescrever_artefato_existente(script, tmp_path, monkeypatch):
    """Relancar sem a flag apagaria horas de execucao na primeira gravacao.

    Foi o que quase aconteceu quando a maquina reiniciou no meio da execucao
    completa: o script recomecaria de lista vazia e regravaria o arquivo com uma
    execucao so. Recusar e o unico comportamento que nao perde dado em silencio.
    """
    execucoes = _duas_execucoes_distintas(script)
    destino = tmp_path / "exec.json"
    _rodar(script, monkeypatch, execucoes[:1], destino)
    antes = destino.read_bytes()

    with pytest.raises(SystemExit):
        _rodar(script, monkeypatch, execucoes, destino)

    assert destino.read_bytes() == antes

"""Testes do script que aplica a curadoria e grava os conjuntos."""

import importlib.util
import json

import pytest

from src import config


@pytest.fixture(scope="module")
def script():
    caminho = config.PROJECT_ROOT / "scripts" / "montar_conjuntos.py"
    spec = importlib.util.spec_from_file_location("montar_conjuntos", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _item(pergunta, resposta, protocolo="PROT-01", ids=("PROT-01 v1 §1",)):
    return {"pergunta": pergunta, "resposta": resposta, "protocolo": protocolo,
            "ids_recebidos": list(ids), "bloco": "[PROT-01 v1 §1]\n## Conduta\nMamografia aos 50 anos.",
            "origem": config.FASE3_ORIGEM_PROFESSOR}


TEMAS_DISTINTOS = ("vacinacao infantil", "pressao arterial", "qualidade do sono", "futebol de varzea",
                   "receita de bolo", "linguagem Python", "eleicoes municipais", "previsao do tempo",
                   "saude bucal", "atividade fisica")


def _gravar(diretorio, nome, itens):
    diretorio.mkdir(parents=True, exist_ok=True)
    (diretorio / f"{nome}.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in itens), encoding="utf-8")


@pytest.fixture
def gerado(tmp_path):
    origem = tmp_path / "gerado"
    _gravar(origem, "perguntas_treino", [
        _item("Com que idade começar o rastreamento?", "Aos 50 anos [PROT-01 v1 §1]. A decisão é do médico."),
        _item("Pode receitar algo?", "Tome 500 mg de 8 em 8 horas [PROT-01 v1 §1]."),
        _item("Qual a idade?", "A mamografia de rastreamento deve ser feita aos 50 anos, mas a resposta nao traz a fonte."),
        _item("Tem fonte inventada?", "Sim, isso esta no protocolo da paciente [PROT-09 v1 §3]."),
    ])
    _gravar(origem, "perguntas_teste", [_item("Quando iniciar a mamografia?", "A mamografia de rastreamento comeca aos 50 anos [PROT-01 v1 §1].")])
    for nome in ("recusas", "fora_do_escopo", "gerais"):
        _gravar(origem, nome, [{"pergunta": f"O que voce sabe sobre {tema}?",
                                "resposta": "Nao posso ajudar com isso, a decisao e do medico.",
                                "protocolo": None, "origem": config.FASE3_ORIGEM_PROFESSOR}
                               for tema in TEMAS_DISTINTOS])
    _gravar(origem, "nao_encontrado", [])
    return origem


def test_reprovados_saem_com_motivo_e_aprovados_viram_exemplos(script, gerado, tmp_path):
    relatorio = script.montar(origem=gerado, destino=tmp_path / "conjuntos")

    motivos = {d["motivo"] for d in relatorio["descartados"]}
    assert {"posologia", "sem citacao valida"} <= motivos
    perguntas_treino = [json.loads(l)["mensagens"][0]["content"]
                        for l in (tmp_path / "conjuntos" / "treino.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any("receitar" in p for p in perguntas_treino)


def test_exemplos_sem_protocolo_tambem_tem_parte_no_teste(script, gerado, tmp_path):
    script.montar(origem=gerado, destino=tmp_path / "conjuntos")
    teste = [json.loads(l) for l in (tmp_path / "conjuntos" / "teste.jsonl").read_text(encoding="utf-8").splitlines()]

    assert {"recusa", "fora_do_escopo", "geral"} <= {e["tipo"] for e in teste}


def test_relatorio_conta_cada_conjunto(script, gerado, tmp_path):
    relatorio = script.montar(origem=gerado, destino=tmp_path / "conjuntos")

    assert set(relatorio["contagens"]) == {"treino", "validacao", "teste"}
    assert sum(relatorio["contagens"].values()) > 0

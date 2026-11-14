"""Aplica a curadoria aos textos do professor e grava os conjuntos.

Roda no container (nao precisa de GPU). Le `data/clinica/gerado/`, grava
`data/clinica/conjuntos/{treino,validacao,teste}.jsonl` e o relatorio em
`results/fase3/curadoria.json`, com cada descarte e o motivo.

    python scripts/montar_conjuntos.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.clinica import curadoria  # noqa: E402
from src.finetune import conjunto  # noqa: E402

TIPOS_SEM_PROTOCOLO = {"recusas": "recusa", "fora_do_escopo": "fora_do_escopo", "gerais": "geral"}


def _ler(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def _motivo(item: dict, exige_citacao: bool) -> str | None:
    if curadoria.tem_posologia(item["resposta"]):
        return "posologia"
    if not curadoria.em_portugues(item["resposta"]):
        return "fora do portugues"
    if exige_citacao and not curadoria.citacoes_validas(item["resposta"], set(item["ids_recebidos"])):
        return "sem citacao valida"
    if exige_citacao and curadoria.numeros_sem_base(item["resposta"], item["bloco"]):
        return "numero sem base nos trechos"
    return None


def _exemplo(item: dict, tipo: str) -> conjunto.Exemplo:
    usuario = conjunto.mensagem_do_usuario(item["pergunta"], item.get("bloco"))
    return conjunto.Exemplo(tipo=tipo, protocolo=item.get("protocolo"), origem=item["origem"],
                            mensagens=[{"role": "user", "content": usuario},
                                       {"role": "assistant", "content": item["resposta"].strip()}])


def _aprovar(itens: list[dict], tipo: str, exige_citacao: bool, descartados: list) -> list[dict]:
    aprovados = []
    for item in itens:
        motivo = _motivo(item, exige_citacao)
        if motivo:
            descartados.append({"tipo": tipo, "pergunta": item["pergunta"], "motivo": motivo})
        else:
            aprovados.append(item)
    unicos, repetidos = curadoria.deduplicar(aprovados, chave="pergunta")
    descartados += [{"tipo": tipo, "pergunta": r["pergunta"], "motivo": r["motivo"]} for r in repetidos]
    return unicos


def montar(*, origem: Path = config.CLINICA_GERADO_DIR,
           destino: Path = config.CLINICA_CONJUNTOS_DIR) -> dict:
    descartados: list[dict] = []
    candidatos, teste = [], []

    for item in _aprovar(_ler(origem / "perguntas_treino.jsonl"), "pergunta", True, descartados):
        candidatos.append(_exemplo(item, "pergunta"))
    for item in _aprovar(_ler(origem / "perguntas_teste.jsonl"), "pergunta", True, descartados):
        teste.append(_exemplo(item, "pergunta"))
    for item in _aprovar(_ler(origem / "nao_encontrado.jsonl"), "nao_encontrado", False, descartados):
        candidatos.append(_exemplo(item, "nao_encontrado"))

    rng = random.Random(config.SEED)
    for nome, tipo in TIPOS_SEM_PROTOCOLO.items():
        aprovados = _aprovar(_ler(origem / f"{nome}.jsonl"), tipo, False, descartados)
        rng.shuffle(aprovados)
        n_teste = max(1, round(len(aprovados) * config.FASE3_FRACAO_TESTE_SEM_PROTOCOLO)) if aprovados else 0
        teste += [_exemplo(i, tipo) for i in aprovados[:n_teste]]
        candidatos += [_exemplo(i, tipo) for i in aprovados[n_teste:]]

    ineditos = {"PROT-07"}
    divisoes = conjunto.dividir(candidatos, teste, ineditos=ineditos, semente=config.SEED)
    for nome in ("treino", "validacao", "teste"):
        conjunto.salvar_jsonl(divisoes[nome], Path(destino) / f"{nome}.jsonl")
    return {
        "contagens": {nome: len(divisoes[nome]) for nome in ("treino", "validacao", "teste")},
        "por_tipo": {nome: {t: sum(e.tipo == t for e in divisoes[nome])
                            for t in sorted({e.tipo for e in divisoes[nome]})}
                     for nome in ("treino", "validacao", "teste")},
        "vazamentos_removidos": divisoes["vazamentos_removidos"],
        "descartados": descartados,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    relatorio = montar()
    destino = config.FASE3_DIR / "curadoria.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("contagens", "vazamentos_removidos")}, ensure_ascii=False))
    print(f"descartados: {len(relatorio['descartados'])}")


if __name__ == "__main__":
    main()

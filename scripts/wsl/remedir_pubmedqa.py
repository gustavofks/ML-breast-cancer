"""Refaz so a etapa PubMedQA da disputa, com a leitura tolerante de resposta.

A primeira rodada comparava o inicio da resposta com yes/no/maybe e contava
`**Yes**` como erro. Esta rodada usa `ler_decisao` e guarda a distribuicao das
respostas brutas, para que a correcao possa ser conferida.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import disputa_modelos as d  # noqa: E402
from src import config  # noqa: E402


def main() -> None:
    dev = d.dividir_pubmedqa().select(range(d.N_PUBMEDQA))
    saida = {}
    for nome in d.CANDIDATOS[:2]:
        tok = AutoTokenizer.from_pretrained(nome)
        modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.bfloat16, device_map={"": 0})
        modelo.eval()
        brutas, acertos = [], 0
        for linha in dev:
            contexto = " ".join(linha["context"]["contexts"])
            pergunta = (f"Context: {contexto}\n\nQuestion: {linha['question']}\n"
                        "Answer with exactly one word: yes, no, or maybe.")
            resposta = d.responder(modelo, tok, pergunta, 8)
            brutas.append(resposta.strip()[:20])
            acertos += d.ler_decisao(resposta) == linha["final_decision"]
        saida[nome] = {"pubmedqa_dev": round(acertos / d.N_PUBMEDQA, 3),
                       "respostas_mais_comuns": collections.Counter(brutas).most_common(6),
                       "sem_decisao": sum(d.ler_decisao(b) is None for b in brutas)}
        print(nome, saida[nome], flush=True)
        del modelo
        torch.cuda.empty_cache()
    gabarito = collections.Counter(linha["final_decision"] for linha in dev)
    saida["distribuicao_do_gabarito"] = dict(gabarito)
    destino = config.FASE3_MEDICOES_DIR / "disputa_pubmedqa_remedido.json"
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

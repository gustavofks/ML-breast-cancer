"""Base contra ajustado, no mesmo servidor: respostas e esquecimento.

Nao e a avaliacao da semana 6 (que usa o teste, intocado ate la). Aqui so a
validacao e o PubMedQA de desenvolvimento, para pegar cedo dois defeitos: o
ajuste nao ter ensinado o formato (citacao, recusa) e o ajuste ter apagado
conhecimento geral. Os dois bracos rodam em GGUF Q8_0 feito pelo mesmo
conversor (servir_assistente.sh), para que a diferenca medida venha do ajuste e
nao da quantizacao.

    python scripts/wsl/checar_ajuste.py   # com servir_assistente.sh no ar
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

import disputa_modelos as disputa  # noqa: E402
from src import config  # noqa: E402
from src.assistente.modelo import ServidorCompativelOpenAI  # noqa: E402
from src.clinica import curadoria  # noqa: E402
from src.corpus import protocolos  # noqa: E402
from src.finetune import conjunto  # noqa: E402
from src.llm.provider import GeracaoIncompleta  # noqa: E402

BRACOS = {"base": "http://localhost:8093", "ajustado": "http://localhost:8092"}
TIPOS = ("pergunta", "recusa", "nao_encontrado", "geral", "fora_do_escopo")
POR_TIPO = 2


def _respostas(cliente, exemplos) -> list[dict]:
    saida = []
    for e in exemplos:
        try:
            texto, cortada = cliente.conversar([e.mensagens[0]], max_tokens=600), False
        except GeracaoIncompleta:
            texto, cortada = "", True
        saida.append({"tipo": e.tipo, "protocolo": e.protocolo, "pergunta": e.pergunta,
                      "resposta": texto, "cortada": cortada,
                      "cita": bool(protocolos.citacoes_em(texto)),
                      "posologia": curadoria.tem_posologia(texto)})
    return saida


def _uma_palavra(url: str, pergunta: str) -> str:
    """Chamada direta, sem o cliente: aqui o corte em 8 tokens e esperado (a
    semana 1 media igual) e a decisao vem nas primeiras palavras; o cliente
    trataria o corte como erro."""
    r = httpx.post(f"{url}/v1/chat/completions", timeout=120, json={
        "messages": [{"role": "user", "content": pergunta}], "max_tokens": 8,
        "temperature": 0.0, "seed": config.SEED})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _pubmedqa(url: str) -> dict:
    dev = disputa.dividir_pubmedqa().select(range(disputa.N_PUBMEDQA))
    brutas, acertos = [], 0
    for linha in dev:
        contexto = " ".join(linha["context"]["contexts"])
        pergunta = (f"Context: {contexto}\n\nQuestion: {linha['question']}\n"
                    "Answer with exactly one word: yes, no, or maybe.")
        resposta = _uma_palavra(url, pergunta)
        brutas.append(resposta.strip()[:20])
        acertos += disputa.ler_decisao(resposta) == linha["final_decision"]
    return {"acuracia": round(acertos / disputa.N_PUBMEDQA, 3),
            "sem_decisao": sum(disputa.ler_decisao(b) is None for b in brutas),
            "respostas_mais_comuns": collections.Counter(brutas).most_common(6)}


def _resumo(respostas: list[dict]) -> dict:
    return {t: {"cita": sum(r["cita"] for r in respostas if r["tipo"] == t),
                "posologia": sum(r["posologia"] for r in respostas if r["tipo"] == t),
                "cortadas": sum(r["cortada"] for r in respostas if r["tipo"] == t)}
            for t in TIPOS}


def main() -> None:
    validacao = conjunto.carregar_jsonl(config.CLINICA_CONJUNTOS_DIR / "validacao.jsonl")
    amostra = [e for t in TIPOS for e in [x for x in validacao if x.tipo == t][:POR_TIPO]]
    resultado = {}
    for braco, url in BRACOS.items():
        respostas = _respostas(ServidorCompativelOpenAI(url, nome=braco), amostra)
        resultado[braco] = {"resumo": _resumo(respostas), "pubmedqa_dev": _pubmedqa(url),
                            "respostas": respostas}
        print(braco, json.dumps(resultado[braco]["resumo"]),
              "pubmedqa_dev", resultado[braco]["pubmedqa_dev"]["acuracia"], flush=True)
    destino = config.FASE3_TREINO_DIR / "checagem.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

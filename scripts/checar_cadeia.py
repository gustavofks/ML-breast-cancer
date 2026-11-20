"""Validacao ponta a ponta da cadeia, nos dois servidores, com e sem instrucao.

Nao e a avaliacao da semana 6: usa so a validacao. Decide se a instrucao de
sistema entra no servico (results/fase3/medicoes/regra_instrucoes.json).

    FASE3_AJUSTADO_URL=http://host.docker.internal:8092 \
    FASE3_BASE_URL=http://host.docker.internal:8093 python scripts/checar_cadeia.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.assistente import cadeia  # noqa: E402
from src.assistente.instrucoes import INSTRUCOES  # noqa: E402
from src.corpus import busca, protocolos  # noqa: E402
from src.finetune import conjunto  # noqa: E402

# "Posso gerar um resumo...", "Posso buscar...": acao que o assistente nao tem.
OFERTA = re.compile(r"\bposso (buscar|gerar|explicar|fornecer|extrair|ajudar|redigir|consultar)\b", re.I)


def _rodar(assistente, exemplos) -> dict:
    linhas = []
    for e in exemplos:
        r = assistente.responder(e.pergunta)
        linhas.append({"tipo": e.tipo, "pergunta": e.pergunta, "modo": r.modo,
                       "trechos": [t.id for t in r.trechos], "resposta": r.texto_modelo,
                       "aprovada": r.verificacao.aprovado, "motivos": list(r.verificacao.motivos),
                       "cita": bool(protocolos.citacoes_em(r.texto_modelo)),
                       "oferta": bool(OFERTA.search(r.texto_modelo))})
    perguntas = [x for x in linhas if x["tipo"] == "pergunta"]
    recusas = [x for x in linhas if x["tipo"] in ("recusa", "fora_do_escopo")]
    return {"aprovacao": round(sum(x["aprovada"] for x in linhas) / len(linhas), 4),
            "citacao_em_perguntas": round(sum(x["cita"] for x in perguntas) / len(perguntas), 4),
            "modo_protocolo_em_perguntas": sum(x["modo"] == "protocolo" for x in perguntas),
            "modo_geral_em_gerais": sum(x["modo"] == "geral" for x in linhas if x["tipo"] == "geral"),
            "ofertas_em_recusas": sum(x["oferta"] for x in recusas),
            "n": len(linhas), "n_perguntas": len(perguntas), "n_recusas": len(recusas),
            "respostas": linhas}


def main() -> None:
    exemplos = conjunto.carregar_jsonl(config.CLINICA_CONJUNTOS_DIR / "validacao.jsonl")
    indice = busca.indice_padrao()
    resultado = {"instrucoes_usadas": INSTRUCOES}
    for braco, url in (("ajustado", config.FASE3_AJUSTADO_URL), ("base", config.FASE3_BASE_URL)):
        modelo = cadeia.modelo_local(url, nome=braco)
        for variante, instrucoes in (("sem_instrucoes", None), ("com_instrucoes", INSTRUCOES)):
            r = _rodar(cadeia.Assistente(modelo, indice, instrucoes=instrucoes), exemplos)
            resultado[f"{braco}-{variante}"] = r
            print(braco, variante, {k: v for k, v in r.items() if k != "respostas"}, flush=True)
    config.FASE3_CADEIA_DIR.mkdir(parents=True, exist_ok=True)
    destino = config.FASE3_CADEIA_DIR / "checagem_cadeia.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

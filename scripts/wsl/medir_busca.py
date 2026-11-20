"""Busca lexical contra vetorial, nos dois modos, e o limiar de pergunta geral.

Roda no WSL: o braco vetorial usa intfloat/multilingual-e5-small com
transformers, que ja esta no ambiente da GPU. Mede em treino e validacao (a
busca nao tem parametro treinado); o teste fica intocado. A regra de escolha foi
gravada antes: results/fase3/medicoes/regra_busca.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402
from src.corpus import busca, protocolos  # noqa: E402
from src.finetune import conjunto  # noqa: E402

E5 = "intfloat/multilingual-e5-small"
K = 4


class BuscaVetorial:
    def __init__(self, lista):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.trechos = list(lista)
        self._tok = AutoTokenizer.from_pretrained(E5)
        self._modelo = AutoModel.from_pretrained(E5).to("cuda").eval()
        self._matriz = self._codificar(["passage: " + t.texto for t in self.trechos])

    def _codificar(self, textos):
        torch = self._torch
        entrada = self._tok(textos, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to("cuda")
        with torch.no_grad():
            saida = self._modelo(**entrada).last_hidden_state
        mascara = entrada["attention_mask"].unsqueeze(-1).float()
        media = (saida * mascara).sum(1) / mascara.sum(1)
        return torch.nn.functional.normalize(media, dim=-1)

    def pontuar(self, pergunta):
        return (self._matriz @ self._codificar(["query: " + pergunta])[0]).tolist()


def _secao(id_trecho: str) -> str:
    return re.sub(r" v\d+", "", id_trecho)


def _perguntas():
    exemplos = [e for nome in ("treino", "validacao")
                for e in conjunto.carregar_jsonl(config.CLINICA_CONJUNTOS_DIR / f"{nome}.jsonl")]
    de_protocolo = [(e.pergunta, {_secao(c) for c in protocolos.citacoes_em(e.mensagens[1]["content"])})
                    for e in exemplos if e.tipo == "pergunta"]
    gerais = [e.pergunta for e in exemplos if e.tipo in ("geral", "fora_do_escopo")]
    return [p for p in de_protocolo if p[1]], gerais


def _limiar(positivas: list[float], negativas: list[float]) -> dict:
    notas = sorted(set(positivas + negativas))
    candidatos = [(a + b) / 2 for a, b in zip(notas, notas[1:])] or notas
    melhor = None
    for limiar in candidatos:
        sens = sum(n >= limiar for n in positivas) / len(positivas)
        espec = sum(n < limiar for n in negativas) / len(negativas)
        balanceada = (sens + espec) / 2
        if melhor is None or balanceada > melhor["acuracia_balanceada"]:
            melhor = {"limiar": round(limiar, 4), "acuracia_balanceada": round(balanceada, 4),
                      "protocolo_reconhecida": round(sens, 4), "geral_reconhecida": round(espec, 4)}
    return melhor


def main() -> None:
    de_protocolo, gerais = _perguntas()
    trechos = busca.trechos_vigentes()
    buscadores = {"lexical": busca.BuscaLexical(trechos), "vetorial": BuscaVetorial(trechos)}
    resultado = {"n_perguntas_protocolo": len(de_protocolo), "n_perguntas_gerais": len(gerais),
                 "k": K, "bracos": {}}
    for tipo, buscador in buscadores.items():
        for modo in ("trechos", "protocolo"):
            acertos, notas_protocolo = 0, []
            for pergunta, gabarito in de_protocolo:
                achados = (busca.melhor_protocolo(buscador, pergunta) if modo == "protocolo"
                           else busca.melhores_trechos(buscador, pergunta, k=K))
                acertos += gabarito <= {_secao(a.trecho.id) for a in achados}
                notas_protocolo.append(max((a.nota for a in achados), default=0.0))
            notas_gerais = [max(buscador.pontuar(p)) for p in gerais]
            resultado["bracos"][f"{tipo}-{modo}"] = {
                "acerto": round(acertos / len(de_protocolo), 4),
                "limiar": _limiar(notas_protocolo, notas_gerais)}
            print(tipo, modo, resultado["bracos"][f"{tipo}-{modo}"], flush=True)
    destino = config.FASE3_MEDICOES_DIR / "busca.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

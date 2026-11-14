"""Disputa dos modelos base da Fase 3, com a regra gravada antes.

Roda no WSL. Para cada candidato: acuracia zero-shot no PubMedQA de
desenvolvimento, idioma das respostas a 20 perguntas de controle em portugues,
e 30 passos de QLoRA com a receita em bfloat16 (tempo e memoria). As perguntas
de controle sao so entrada de avaliacao: as respostas sao medidas e descartadas,
nunca viram treino. A regra esta em results/fase3/medicoes/regra_disputa.json.
"""

from __future__ import annotations

import gc
import json
import random
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src import config  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

CANDIDATOS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it"]
N_PUBMEDQA = 200
CONTROLE = [
    "O que e mamografia de rastreamento?", "Quando a ultrassonografia mamaria e indicada?",
    "O que significa BI-RADS 3?", "O que e uma PAAF de mama?", "Quais sinais de alerta na mama?",
    "Qual a diferenca entre nodulo benigno e maligno?", "O que e biopsia por agulha grossa?",
    "Para que serve o exame clinico das mamas?", "O que e densidade mamaria?",
    "Quando repetir um exame de imagem inconclusivo?", "O que e BI-RADS 0?",
    "O que e mastalgia ciclica?", "Por que historico familiar importa?",
    "O que e um cisto mamario simples?", "O que e calcificacao na mamografia?",
    "O que e laudo citopatologico?", "O que fazer com exame pendente ha muito tempo?",
    "O que e falso negativo em triagem?", "Por que o medico tem a palavra final?",
    "O que e retorno de seguimento?",
]
_PALAVRAS_PT = {"que", "nao", "não", "uma", "para", "com", "mama", "exame", "pode", "ser",
                "dos", "das", "é", "são"}


def dividir_pubmedqa():
    dados = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    indices = list(range(len(dados)))
    random.Random(config.SEED).shuffle(indices)
    return dados.select(indices[500:])  # desenvolvimento; indices[:500] e o teste final


def em_portugues(texto: str) -> bool:
    palavras = re.findall(r"[a-zà-ú]+", texto.lower())
    return sum(p in _PALAVRAS_PT for p in palavras) >= 3


def responder(modelo, tok, mensagem: str, n: int) -> str:
    # enable_thinking=False: o Qwen3.5 pensa por padrao, e o pensamento ocuparia
    # os poucos tokens da resposta de uma palavra. Os outros moldes ignoram.
    entrada = tok.apply_chat_template([{"role": "user", "content": mensagem}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to("cuda")
    n_entrada = entrada["input_ids"].shape[1]
    with torch.no_grad():
        saida = modelo.generate(**entrada, max_new_tokens=n, do_sample=False)
    return tok.decode(saida[0, n_entrada:], skip_special_tokens=True)


def acuracia_pubmedqa(modelo, tok, dados) -> float:
    acertos = 0
    for linha in dados.select(range(N_PUBMEDQA)):
        contexto = " ".join(linha["context"]["contexts"])
        pergunta = (f"Context: {contexto}\n\nQuestion: {linha['question']}\n"
                    "Answer with exactly one word: yes, no, or maybe.")
        acertos += ler_decisao(responder(modelo, tok, pergunta, 8)) == linha["final_decision"]
    return acertos / N_PUBMEDQA


def ler_decisao(resposta: str) -> str | None:
    """Primeira ocorrencia de yes, no ou maybe, ignorando formatacao.

    Comparar o inicio da resposta penalizava quem responde com markdown
    (`**Yes**`): a primeira rodada da disputa mediu formato, nao conhecimento.
    """
    achado = re.search(r"\b(yes|no|maybe)\b", resposta.lower())
    return achado.group(1) if achado else None


def treino_curto(nome: str) -> dict:
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModelForCausalLM.from_pretrained(nome, quantization_config=q,
                                                  device_map={"": 0}, dtype=torch.bfloat16)
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.enable_input_require_grads()
    modelo.config.use_cache = False
    modelo = get_peft_model(modelo, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                               task_type="CAUSAL_LM"), autocast_adapter_dtype=False)
    modelo.train()
    otimizador = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=2e-4)
    texto = tok.apply_chat_template([{"role": "user", "content": CONTROLE[0]},
                                     {"role": "assistant", "content": "Resposta de teste. " * 60}],
                                    tokenize=False, enable_thinking=False)
    lote = tok([texto] * 4, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    inicio = None
    for passo in range(30):
        if passo == 3:
            torch.cuda.synchronize()
            inicio = time.time()
        perda = modelo(**lote, labels=lote.input_ids).loss
        perda.backward()
        otimizador.step()
        otimizador.zero_grad()
    torch.cuda.synchronize()
    resultado = {"s_por_passo": round((time.time() - inicio) / 27, 2),
                 "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "perda_final": round(perda.item(), 3)}
    del modelo
    return resultado


def avaliar(nome: str, dev) -> dict:
    resultado = {"modelo": nome}
    try:
        tok = AutoTokenizer.from_pretrained(nome)
        modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.bfloat16, device_map={"": 0})
        modelo.eval()
        respostas = [responder(modelo, tok, p, 120) for p in CONTROLE]
        resultado["portugues"] = sum(em_portugues(r) for r in respostas)
        resultado["amostra"] = respostas[2][:300]
        resultado["pubmedqa_dev"] = round(acuracia_pubmedqa(modelo, tok, dev), 3)
        del modelo
        gc.collect()
        torch.cuda.empty_cache()
        resultado["treino"] = treino_curto(nome)
    except Exception as erro:  # eliminatorio: registrar e seguir para o proximo
        resultado["erro"] = f"{type(erro).__name__}: {str(erro)[:300]}"
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(resultado, ensure_ascii=False), flush=True)
    return resultado


def main() -> None:
    dev = dividir_pubmedqa()
    resultados = [avaliar(nome, dev) for nome in CANDIDATOS]
    destino = config.FASE3_MEDICOES_DIR / "disputa.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resultados, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

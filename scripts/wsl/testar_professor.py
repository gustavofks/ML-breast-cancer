"""Teste do modelo professor: memoria, velocidade e uma amostra de protocolo.

O professor escreve os textos do hospital ficticio que vao para o treino, no
lugar do Claude (clausula D.4 dos termos da Anthropic). A amostra gerada aqui e
so para conferencia e nao entra em nenhum conjunto.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src import config  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

PROFESSOR = "Qwen/Qwen3.5-9B"
PEDIDO = (
    f"Voce escreve documentos internos do {config.CLINICA_NOME}, um hospital ficticio. "
    "Escreva a secao 'Conduta' de um protocolo interno para mamografia classificada como "
    "BI-RADS 3, com base nestes fatos: achado provavelmente benigno; risco de malignidade "
    "menor que 2%; controle radiologico em 6 meses; depois aos 12 e 24 meses; achado que "
    "cresce ou muda de aspecto passa a ser investigado com biopsia. Use linguagem tecnica "
    "e itens numerados. Nao invente doses nem medicamentos."
)


def main() -> None:
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(PROFESSOR)
    modelo = AutoModelForCausalLM.from_pretrained(PROFESSOR, quantization_config=q,
                                                  device_map={"": 0}, dtype=torch.bfloat16)
    entrada = tok.apply_chat_template([{"role": "user", "content": PEDIDO}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to("cuda")
    n_entrada = entrada["input_ids"].shape[1]
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    with torch.no_grad():
        saida = modelo.generate(**entrada, max_new_tokens=400, do_sample=False)
    torch.cuda.synchronize()
    gerados = saida.shape[1] - n_entrada
    resultado = {
        "modelo": PROFESSOR, "quantizacao": "nf4",
        "tokens_s": round(gerados / (time.time() - inicio), 1),
        "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "amostra": tok.decode(saida[0, n_entrada:], skip_special_tokens=True),
    }
    destino = config.FASE3_MEDICOES_DIR / "professor.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

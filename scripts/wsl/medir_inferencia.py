"""Mede a velocidade de geracao do modelo base em cada servidor candidato.

Roda no WSL (`fase3-gpu`). A regra de escolha esta no plano da semana 1 e foi
gravada antes de medir. Saida: results/fase3/medicoes/inferencia.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src import config  # noqa: E402

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
LLAMA = Path("/root/gpu/llama.cpp")
GGUF_DIR = Path("/root/gpu/gguf")
PERGUNTA = "Explique em portugues, em um paragrafo, o que e rastreamento mamografico."
N_TOKENS = 200


def medir_transformers() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map={"": 0})
    entrada = tok.apply_chat_template([{"role": "user", "content": PERGUNTA}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("cuda")
    n_entrada = entrada["input_ids"].shape[1]
    modelo.generate(**entrada, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    inicio = time.time()
    saida = modelo.generate(**entrada, max_new_tokens=N_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    gerados = saida.shape[1] - n_entrada
    texto = tok.decode(saida[0, n_entrada:], skip_special_tokens=True)
    resultado = {"tokens_s": round(gerados / (time.time() - inicio), 1), "tokens": int(gerados),
                 "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "amostra": texto[:300]}
    del modelo
    torch.cuda.empty_cache()
    return resultado


def medir_llama(quantizacao: str) -> dict:
    from huggingface_hub import snapshot_download

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    f16 = GGUF_DIR / "base-f16.gguf"
    alvo = GGUF_DIR / f"base-{quantizacao}.gguf"
    if not f16.exists():
        origem = snapshot_download(MODELO)
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), origem,
                        "--outfile", str(f16), "--outtype", "f16"], check=True,
                       capture_output=True)
    if not alvo.exists():
        subprocess.run([str(LLAMA / "build/bin/llama-quantize"), str(f16), str(alvo), quantizacao],
                       check=True, capture_output=True)
    bench = subprocess.run(
        [str(LLAMA / "build/bin/llama-bench"), "-m", str(alvo), "-p", "512", "-n", str(N_TOKENS),
         "-ngl", "99", "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    linhas = json.loads(bench.stdout)
    geracao = next(linha for linha in linhas if linha["n_gen"] > 0)
    leitura = next(linha for linha in linhas if linha["n_prompt"] > 0)
    return {"tokens_s": round(geracao["avg_ts"], 1), "leitura_tokens_s": round(leitura["avg_ts"], 1),
            "arquivo_gb": round(alvo.stat().st_size / 2**30, 2)}


def main() -> None:
    destino = config.FASE3_MEDICOES_DIR / "inferencia.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado = {"modelo": MODELO, "pergunta": PERGUNTA, "data": time.strftime("%Y-%m-%d")}
    resultado["transformers_bf16"] = medir_transformers()
    if (LLAMA / "build/bin/llama-bench").exists():
        for quantizacao in ("Q8_0", "Q4_K_M"):
            try:
                resultado[f"llama_cpp_{quantizacao}"] = medir_llama(quantizacao)
            except Exception as erro:  # registrar e seguir: a reserva ja esta medida
                detalhe = getattr(erro, "stderr", None) or str(erro)
                resultado[f"llama_cpp_{quantizacao}"] = {"erro": str(detalhe)[-300:]}
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Mescla o adaptador LoRA no modelo base em bfloat16, para converter em GGUF.

Por que mesclar em vez de servir o adaptador com --lora: na conversao do
Qwen3.5, o llama.cpp funde matrizes da atencao linear (in_proj_qkv com
in_proj_z em attn_qkv_gate, in_proj_b com in_proj_a em ssm_beta_alpha) e
reordena as cabecas V. O conversor de LoRA nao repete essas transformacoes, e o
servidor recusou o adaptador ("LoRA tensor 'blk.0.attn_gate.weight' does not
exist in base model"). Mesclado, o modelo ajustado passa pelo mesmo conversor
que gerou o GGUF base.

A mescla e feita sobre os pesos bfloat16 originais, nao sobre os NF4 do treino:
e o procedimento padrao do QLoRA, e a mesma aproximacao que o --lora sobre o
GGUF Q8_0 faria.

    python scripts/wsl/mesclar_adaptador.py /root/gpu/adaptadores/<nome>/melhor /root/gpu/mesclado
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adaptador", type=Path)
    parser.add_argument("destino", type=Path)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Mesma classe do treino (AutoModelForCausalLM): os nomes do adaptador
    # (model.layers.N...) so batem com ela.
    base = AutoModelForCausalLM.from_pretrained(config.FASE3_MODELO_BASE, dtype=torch.bfloat16,
                                                device_map={"": 0})
    modelo = PeftModel.from_pretrained(base, str(args.adaptador)).merge_and_unload()
    restantes = [n for n, _ in modelo.named_modules() if "lora" in n.lower()]
    assert not restantes, f"camadas LoRA sobraram depois da mescla: {restantes[:3]}"
    args.destino.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(args.destino, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.FASE3_MODELO_BASE).save_pretrained(args.destino)
    print("mesclado em", args.destino, flush=True)


if __name__ == "__main__":
    main()

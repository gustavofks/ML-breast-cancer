"""Fine-tuning QLoRA do modelo base com TRL, no WSL.

    python scripts/wsl/treinar.py --ensaio --adaptador bf16 --lote 1   # 20 passos, mede tempo
    python scripts/wsl/treinar.py --adaptador fp32            # treino completo

O modelo chega ao SFTTrainer ja embrulhado com LoRA e sem peft_config: com
peft_config, o TRL chamaria prepare_model_for_kbit_training, que converte a
camada final para float32 (39 vezes mais lento nesta placa, semana 1).
Historico, receita e melhor epoca vao para results/fase3/treino/; o adaptador,
grande demais para o Git, vai para /root/gpu/adaptadores/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402
from src.finetune import conjunto, receita  # noqa: E402

ADAPTADORES = Path("/root/gpu/adaptadores")


def _renderizador(tok):
    def renderizar(mensagens, *, gerar):
        return tok.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=gerar,
                                       enable_thinking=False)
    return renderizar


def _dataset(caminho, renderizar, limite=None):
    from datasets import Dataset

    exemplos = conjunto.carregar_jsonl(caminho)[:limite]
    return Dataset.from_list([receita.para_prompt_completion(asdict(e), renderizar) for e in exemplos])


def _modelo(r, adaptador_fp32):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    modelo = AutoModelForCausalLM.from_pretrained(r.modelo_base, quantization_config=q,
                                                  device_map={"": 0}, dtype=torch.bfloat16)
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.enable_input_require_grads()
    modelo.config.use_cache = False
    modelo = get_peft_model(modelo, LoraConfig(
        r=r.lora_r, lora_alpha=r.lora_alpha, lora_dropout=r.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM"), autocast_adapter_dtype=adaptador_fp32)
    cabeca = modelo.get_output_embeddings().weight
    assert cabeca.dtype == torch.bfloat16, f"camada final em {cabeca.dtype}: treino ficaria lento"
    return modelo


def _fixar_tipo_do_adaptador(modelo, tipo) -> None:
    """Devolve ao adaptador o tipo pedido, depois do SFTTrainer.

    Com modelo base quantizado, o SFTTrainer converte todo parametro treinavel
    para bfloat16 por conta propria, ignorando o autocast_adapter_dtype: o
    primeiro ensaio "fp32" rodou em bfloat16 e mediu o mesmo pico de memoria. O
    otimizador so nasce em train(), entao converter aqui basta.
    """
    for parametro in modelo.parameters():
        if parametro.requires_grad:
            parametro.data = parametro.data.to(tipo)
    tipos = {p.dtype for p in modelo.parameters() if p.requires_grad}
    assert tipos == {tipo}, f"adaptador em {tipos}, pedido {tipo}"


def _progresso():
    """Uma linha por registro, com hora: o video mostra o treino acontecendo."""
    from transformers import TrainerCallback

    class Progresso(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                campos = {k: round(v, 4) if isinstance(v, float) else v for k, v in logs.items()}
                print(time.strftime("%H:%M:%S"), f"passo {state.global_step}/{state.max_steps}",
                      json.dumps(campos), flush=True)
    return Progresso()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ensaio", action="store_true", help="20 passos em 32 exemplos, sem salvar")
    parser.add_argument("--adaptador", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--lote", type=int, default=4,
                        help="exemplos por passada; a acumulacao se ajusta para manter o lote efetivo")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    r = receita.receita_padrao()
    tok = AutoTokenizer.from_pretrained(r.modelo_base)
    renderizar = _renderizador(tok)
    treino = _dataset(config.CLINICA_CONJUNTOS_DIR / "treino.jsonl", renderizar, 32 if args.ensaio else None)
    validacao = _dataset(config.CLINICA_CONJUNTOS_DIR / "validacao.jsonl", renderizar,
                         8 if args.ensaio else None)

    nome = (time.strftime("%Y%m%d-%H%M") + f"-{args.adaptador}-lote{args.lote}"
            + ("-ensaio" if args.ensaio else ""))
    saida = ADAPTADORES / nome
    modelo = _modelo(r, adaptador_fp32=args.adaptador == "fp32")
    cfg = SFTConfig(
        output_dir=str(saida), seed=r.semente, report_to="none",
        num_train_epochs=r.epocas, max_steps=20 if args.ensaio else -1,
        # Lote efetivo fixo (r.acumulacao exemplos por atualizacao), dividido entre
        # passadas de `lote` exemplos: o Qwen3.5 sem nucleos otimizados paga um custo
        # fixo alto por passada. Ensaio: lote 1 a 70 s por passo, lote 4 a 37 s, com a
        # mesma curva de perda (results/fase3/treino/ensaio-bf16-lote*.json).
        per_device_train_batch_size=args.lote, per_device_eval_batch_size=args.lote,
        gradient_accumulation_steps=max(1, r.acumulacao // args.lote), learning_rate=r.taxa,
        # transformers 5: warmup_steps aceita fracao em [0, 1); warmup_ratio saiu.
        lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.0,
        max_length=r.max_tokens, completion_only_loss=True, packing=False,
        gradient_checkpointing=False,  # ja ligado no modelo, sem reentrada
        bf16=False, fp16=False,  # pesos ja em bfloat16; autocast travou na semana 1
        logging_steps=1 if args.ensaio else 5,
        eval_strategy="no" if args.ensaio else "epoch",
        save_strategy="no" if args.ensaio else "epoch",
        load_best_model_at_end=not args.ensaio, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=r.epocas, dataloader_num_workers=0,
    )
    treinador = SFTTrainer(model=modelo, args=cfg, train_dataset=treino,
                           eval_dataset=None if args.ensaio else validacao,
                           processing_class=tok, callbacks=[_progresso()])
    _fixar_tipo_do_adaptador(treinador.model, torch.float32 if args.adaptador == "fp32" else torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    treinador.train()
    duracao = time.time() - inicio

    resumo = {
        "nome": nome, "receita": asdict(r), "adaptador": args.adaptador, "ensaio": args.ensaio,
        "lote": args.lote,
        "exemplos_treino": len(treino), "exemplos_validacao": len(validacao),
        "duracao_s": round(duracao, 1),
        "s_por_passo": round(duracao / max(1, treinador.state.global_step), 2),
        "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "historico": treinador.state.log_history,
        "melhor_checkpoint": treinador.state.best_model_checkpoint,
        "melhor_perda_validacao": treinador.state.best_metric,
    }
    if not args.ensaio:
        final = saida / "melhor"
        treinador.model.save_pretrained(final)
        resumo["adaptador_dir"] = str(final)
        resumo["adaptador_sha256"] = hashlib.sha256(
            (final / "adapter_model.safetensors").read_bytes()).hexdigest()
    arquivo = f"ensaio-{args.adaptador}-lote{args.lote}.json" if args.ensaio else "treino.json"
    destino = config.FASE3_TREINO_DIR / arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: resumo[k] for k in ("duracao_s", "s_por_passo", "pico_vram_gb",
                                              "melhor_perda_validacao")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

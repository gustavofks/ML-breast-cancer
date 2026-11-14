#!/usr/bin/env bash
# Converte o modelo professor para GGUF Q8_0 (uma vez) e o serve no llama.cpp.
# Quatro vagas paralelas: a geracao em lote manda varios pedidos de uma vez, e o
# servidor os processa juntos.
#
#   wsl -d fase3-gpu -u root -- bash /mnt/c/workspace/ML-breast-cancer/scripts/wsl/servir_professor.sh
set -euo pipefail
export HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm/lib
. /root/gpu/venv/bin/activate
LLAMA=/root/gpu/llama.cpp
GGUF=/root/gpu/gguf
mkdir -p "$GGUF" /root/gpu/logs
if [ ! -f "$GGUF/professor-Q8_0.gguf" ]; then
  ORIGEM=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-9B"))')
  python "$LLAMA/convert_hf_to_gguf.py" "$ORIGEM" --outfile "$GGUF/professor-f16.gguf" --outtype f16
  "$LLAMA/build/bin/llama-quantize" "$GGUF/professor-f16.gguf" "$GGUF/professor-Q8_0.gguf" Q8_0
  rm "$GGUF/professor-f16.gguf"
fi
pkill -f "llama-server.*8091" || true
nohup "$LLAMA/build/bin/llama-server" -m "$GGUF/professor-Q8_0.gguf" -ngl 99 --port 8091 \
  --host 0.0.0.0 -c 32768 -np 4 --jinja --chat-template-kwargs '{"enable_thinking": false}' \
  > /root/gpu/logs/professor.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8091/health | grep -q ok && { echo "professor pronto"; exit 0; }; sleep 2; done
echo "professor nao subiu; ver /root/gpu/logs/professor.log" >&2; exit 1

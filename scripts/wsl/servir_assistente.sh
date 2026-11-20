#!/usr/bin/env bash
# Serve os dois bracos da comparacao no llama.cpp: o modelo ajustado (porta 8092)
# e o base (porta 8093), ambos em GGUF Q8_0 gerado pelo mesmo conversor.
#
# O plano era um servidor so, com --lora e escala 0 para o base. O llama.cpp
# recusou o adaptador convertido: na conversao do Qwen3.5 ele funde matrizes da
# atencao linear e reordena cabecas, e o conversor de LoRA nao faz o mesmo (ver
# mesclar_adaptador.py). O adaptador e mesclado, convertido e quantizado como o
# base; os dois cabem juntos na placa (cerca de 5 GB cada).
#
#   bash scripts/wsl/servir_assistente.sh /root/gpu/adaptadores/<nome>/melhor
#   bash scripts/wsl/servir_assistente.sh --so-servir   # GGUF do ajustado ja pronto
set -euo pipefail
export HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm/lib
. /root/gpu/venv/bin/activate
RAIZ=$(cd "$(dirname "$0")/../.." && pwd)
LLAMA=/root/gpu/llama.cpp
GGUF=/root/gpu/gguf
MESCLADO=/root/gpu/mesclado
ADAPTADOR_HF="$1"
mkdir -p /root/gpu/logs
[ -f "$GGUF/q35-Q8_0.gguf" ] || { echo "falta o GGUF base da semana 1" >&2; exit 1; }

if [ "$ADAPTADOR_HF" != "--so-servir" ]; then
  python "$RAIZ/scripts/wsl/mesclar_adaptador.py" "$ADAPTADOR_HF" "$MESCLADO"
  python "$LLAMA/convert_hf_to_gguf.py" "$MESCLADO" --outfile "$GGUF/ajustado-f16.gguf" --outtype f16
  "$LLAMA/build/bin/llama-quantize" "$GGUF/ajustado-f16.gguf" "$GGUF/ajustado-Q8_0.gguf" Q8_0
  rm -rf "$GGUF/ajustado-f16.gguf" "$MESCLADO"
fi
[ -f "$GGUF/ajustado-Q8_0.gguf" ] || { echo "falta o GGUF do ajustado" >&2; exit 1; }

servir() {  # servir <gguf> <porta> <log>
  pkill -f "llama-server.*--port $2" || true
  nohup "$LLAMA/build/bin/llama-server" -m "$1" -ngl 99 --port "$2" --host 0.0.0.0 \
    -c 16384 -np 2 --jinja --chat-template-kwargs '{"enable_thinking": false}' \
    > "/root/gpu/logs/$3.log" 2>&1 &
}
servir "$GGUF/ajustado-Q8_0.gguf" 8092 assistente-ajustado
servir "$GGUF/q35-Q8_0.gguf" 8093 assistente-base
for porta in 8092 8093; do
  for i in $(seq 1 90); do curl -s "localhost:$porta/health" | grep -q ok && break; sleep 2; done
  curl -s "localhost:$porta/health" | grep -q ok || { echo "porta $porta nao subiu; ver /root/gpu/logs/" >&2; exit 1; }
done
echo "assistente pronto: ajustado em 8092, base em 8093"

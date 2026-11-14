#!/usr/bin/env bash
# Recria o ambiente de treino da Fase 3 dentro da distro WSL `fase3-gpu`.
#
# Por que WSL: o Smart App Control do Windows bloqueia binarios Python sem
# assinatura reconhecida, e no Linux do WSL ele nao atua. Por que estas versoes:
# sao as testadas com a RX 9060 XT (gfx1200) em 2026-11-05.
#
#   wsl --install Ubuntu-24.04 --name fase3-gpu --no-launch
#   wsl -d fase3-gpu -u root -- bash /mnt/c/workspace/ML-breast-cancer/scripts/wsl/preparar_ambiente.sh
#
# Para desfazer tudo: wsl --unregister fase3-gpu
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
ROCM=7.2.4
RAIZ=/root/gpu
REPO=/mnt/c/workspace/ML-breast-cancer

apt-get update -qq
apt-get install -y -qq python3-venv python3-pip python3-dev git curl wget build-essential cmake ca-certificates gnupg

mkdir -p "$RAIZ" && cd "$RAIZ"
if ! dpkg -s amdgpu-install >/dev/null 2>&1; then
  wget -q "https://repo.radeon.com/amdgpu-install/${ROCM}/ubuntu/noble/amdgpu-install_${ROCM}.70204-1_all.deb"
  apt-get install -y -qq "./amdgpu-install_${ROCM}.70204-1_all.deb"
  apt-get update -qq
fi
# Sem o driver do kernel (no WSL quem fala com a placa e o driver do Windows):
# so o runtime e as bibliotecas que o torch "lw" da AMD carrega do sistema.
apt-get install -y -qq --no-install-recommends rocm-hip-runtime rocminfo rocm-libs miopen-hip roctracer rccl

# Ponte entre o ROCm e o driver do Windows.
if ! dpkg -s rocdxg-roct >/dev/null 2>&1; then
  wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
  dpkg -i rocdxg-roct_1.2.2_amd64.deb
fi

grep -q HSA_ENABLE_DXG_DETECTION /root/.bashrc || cat >> /root/.bashrc <<'EOF'
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm/lib
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export TOKENIZERS_PARALLELISM=false
EOF

[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q -U pip wheel
B="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM}"
pip install -q "$B/torch-2.9.1%2Brocm${ROCM}.lw.git39497456-cp312-cp312-linux_x86_64.whl" \
               "$B/triton-3.5.1%2Brocm${ROCM}.gita272dfa8-cp312-cp312-linux_x86_64.whl"
pip list --format=freeze | grep -E '^(torch|triton)==' > pins.txt
pip install -q -r "$REPO/requirements-finetune.txt" -c pins.txt

export HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm/lib
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "GPU nao encontrada"
print("ok:", torch.cuda.get_device_name(0), torch.__version__)
EOF

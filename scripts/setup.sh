#!/usr/bin/env bash
# 一键环境搭建 + 冒烟测试
# 用法: bash scripts/setup.sh   (在项目根目录下执行)
set -e

echo "=== [1/5] 创建虚拟环境(建在 Linux 家目录,避开 /mnt/c 兼容问题)==="
VENV="$HOME/.venvs/rec"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

echo "=== [2/5] 安装依赖 ==="
pip install -q -r requirements.txt
pip install -q torch

echo "=== [3/5] 检查 GPU ==="
python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda_available={ok}")
if ok:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("!! CUDA 不可用:请确认 Windows 侧 NVIDIA 驱动为最新(WSL2 无需单独装 CUDA toolkit)")
EOF

echo "=== [4/5] 单元测试 ==="
python tests/test_recall_dataset.py

echo "=== [5/5] 合成数据冒烟测试(约2-5分钟)==="
python tests/gen_synth_data.py
python -m src.data.preprocess --config configs/synth.yaml
python -m src.data.build_samples --config configs/synth.yaml
python -m src.run_two_tower --config configs/synth.yaml

echo ""
echo "✅ 环境就绪。下一步(Pure 真实数据):"
echo "   source ~/.venvs/rec/bin/activate"
echo "   bash scripts/download_data.sh pure"
echo "   python -m src.data.preprocess --config configs/pure.yaml && \\"
echo "   python -m src.data.eda --config configs/pure.yaml && \\"
echo "   python -m src.data.build_samples --config configs/pure.yaml && \\"
echo "   python -m src.run_baselines --config configs/pure.yaml && \\"
echo "   python -m src.run_two_tower --config configs/pure.yaml"

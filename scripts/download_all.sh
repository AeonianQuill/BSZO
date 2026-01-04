#!/bin/bash
# =============================================================================
# Download OPT-13B model and datasets
# Run this in AIcoder (with internet access)
# Uses ModelScope for model, hf-mirror for datasets
# =============================================================================

# Set cache directories to data disk
export PIP_CACHE_DIR=/mnt/innovator/chl/pip_cache
export TMPDIR=/mnt/innovator/chl/tmp
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets

# Create directories
mkdir -p /mnt/innovator/chl/models
mkdir -p /mnt/innovator/chl/huggingface/datasets
mkdir -p /mnt/innovator/chl/pip_cache
mkdir -p /mnt/innovator/chl/tmp

echo "============================================="
echo "Model dir: /mnt/innovator/chl/models"
echo "Dataset dir: $HF_DATASETS_CACHE"
echo "============================================="

# Download OPT-13B model via ModelScope
echo ""
echo "[1/6] Downloading OPT-13B model via ModelScope..."
modelscope download --model facebook/opt-13b --local_dir /mnt/innovator/chl/models/opt-13b

# Download datasets via hf-mirror
echo ""
echo "[2/6] Downloading SST-2..."
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2'); print('SST-2 done!')"

echo ""
echo "[3/6] Downloading RTE..."
python -c "from datasets import load_dataset; load_dataset('super_glue', 'rte'); print('RTE done!')"

echo ""
echo "[4/6] Downloading WSC..."
python -c "from datasets import load_dataset; load_dataset('super_glue', 'wsc'); print('WSC done!')"

echo ""
echo "[5/6] Downloading COPA..."
python -c "from datasets import load_dataset; load_dataset('super_glue', 'copa'); print('COPA done!')"

echo ""
echo "[6/6] Downloading MultiRC..."
python -c "from datasets import load_dataset; load_dataset('super_glue', 'multirc'); print('MultiRC done!')"

echo ""
echo "============================================="
echo "All downloads complete!"
echo "Model: /mnt/innovator/chl/models/opt-13b"
echo "Datasets: $HF_DATASETS_CACHE"
echo "============================================="

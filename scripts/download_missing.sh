#!/bin/bash
# =============================================================================
# Download missing datasets: wsc.fixed, trec, wic
# Run this with internet access
# =============================================================================

# Set cache directories
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets

echo "============================================="
echo "Downloading missing datasets"
echo "Dataset dir: $HF_DATASETS_CACHE"
echo "============================================="

echo ""
echo "[1/7] Downloading WSC.fixed (SuperGLUE)..."
python -c 'from datasets import load_dataset; load_dataset("super_glue", "wsc.fixed"); print("WSC.fixed done")'

echo ""
echo "[2/7] Downloading TREC..."
python -c 'from datasets import load_dataset; load_dataset("trec"); print("TREC done")'

echo ""
echo "[3/7] Downloading WIC (SuperGLUE)..."
python -c 'from datasets import load_dataset; load_dataset("super_glue", "wic"); print("WIC done")'

echo ""
echo "[4/7] Downloading BoolQ dataset from ModelScope..."
modelscope download --dataset google/boolq --local_dir /mnt/innovator/chl/huggingface/datasets/boolq

echo ""
echo "[5/7] Downloading RoBERTa-Large model from ModelScope..."
modelscope download --model FacebookAI/roberta-large --local_dir /mnt/innovator/chl/models/roberta-large

echo ""
echo "[6/7] Downloading OPT-1.3B model..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'facebook/opt-1.3b'
save_path = '/mnt/innovator/chl/models/opt-1.3b'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f'OPT-1.3B saved to {save_path}')
"

echo ""
echo "[7/7] Downloading Mistral-7B model from ModelScope..."
python -c "
from modelscope import snapshot_download
model_name = 'mistralai/Mistral-7B-v0.1'
save_path = '/mnt/innovator/chl/models/mistral-7b'
print(f'Downloading {model_name} from ModelScope...')
snapshot_download(model_name, local_dir=save_path)
print(f'Mistral-7B saved to {save_path}')
"

echo ""
echo "============================================="
echo "All downloads complete!"
echo "Datasets: $HF_DATASETS_CACHE"
echo "Models: /mnt/innovator/chl/models/"
echo "============================================="

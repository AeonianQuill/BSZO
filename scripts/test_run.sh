#!/bin/bash
# =============================================================================
# Quick test script - verify training works before running full experiments
# Only runs 100 steps on SST2 with zo_sgd
# =============================================================================

export WANDB_MODE=offline
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT="test_run"

MODEL_NAME="/mnt/innovator/chl/models/opt-13b"

echo "============================================="
echo "TEST RUN: ZO-SGD on SST2 (100 steps)"
echo "============================================="

python run.py \
    --model_name=${MODEL_NAME} \
    --task_name=SST2 \
    --output_dir=result/test_run \
    --num_train_epochs=1 \
    --per_device_train_batch_size=16 \
    --load_best_model_at_end \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --save_total_limit=1 \
    --max_steps=100 \
    --logging_steps=10 \
    --num_eval=100 \
    --num_train=100 \
    --num_dev=50 \
    --train_as_classification \
    --perturbation_mode=two_side \
    --trainer=zo_sgd \
    --train_set_seed=0 \
    --lr_scheduler_type=constant \
    --eval_steps=50 \
    --save_steps=50 \
    --learning_rate=1e-6 \
    --zo_eps=1e-4 \
    --weight_decay=0 \
    --load_bfloat16

echo "============================================="
echo "TEST COMPLETE!"
echo ""
echo "If successful, check:"
echo "1. ./wandb/ folder exists"
echo "2. ./result/test_run/ has output"
echo ""
echo "Then sync to wandb:"
echo "  export WANDB_API_KEY=your_key"
echo "  wandb sync ./wandb/"
echo "============================================="

#!/bin/bash
# =============================================================================
# Mistral-7B Memory Usage Test
# Dataset: SST2
# Methods: zo_sgd, zo_adam, hizoo, bszo_v4
# Purpose: Record GPU memory usage for each method (600 steps)
# =============================================================================

# Wandb online mode to record memory
export WANDB_MODE=online
export WANDB_DIR=/mnt/innovator/chl/wandb_cache
export WANDB_CACHE_DIR=/mnt/innovator/chl/wandb_cache

# Set cache directories
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

# Common settings
MODEL_NAME="/mnt/innovator/chl/models/mistral-7b"
TASK="SST2"
BATCH_SIZE=16
MAX_STEPS=600
EVAL_STEPS=200
SAVE_STEPS=200
ZO_EPS="1e-4"
NUM_TRAIN=1000
NUM_DEV=500

echo "============================================="
echo "Mistral-7B Memory Usage Test"
echo "Dataset: ${TASK}"
echo "Steps: ${MAX_STEPS}"
echo "Single GPU mode (CUDA_VISIBLE_DEVICES=0)"
echo "============================================="

# =============================================================================
# ZO-SGD (MeZO)
# =============================================================================
run_zo_sgd() {
    local lr="1e-6"
    echo "======================================"
    echo "[1/4] Running ZO-SGD (MeZO) with lr=${lr}"
    echo "======================================"

    export WANDB_PROJECT="mistral7b_memory_test"

    python run.py \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/zo_sgd_${TASK} \
        --num_train_epochs=5 \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --load_best_model_at_end \
        --evaluation_strategy=steps \
        --save_strategy=steps \
        --save_total_limit=1 \
        --max_steps=${MAX_STEPS} \
        --logging_steps=10 \
        --num_eval=1000 \
        --num_train=${NUM_TRAIN} \
        --num_dev=${NUM_DEV} \
        --train_as_classification \
        --perturbation_mode=two_side \
        --trainer=zo_sgd \
        --train_set_seed=0 \
        --lr_scheduler_type=constant \
        --eval_steps=${EVAL_STEPS} \
        --save_steps=${SAVE_STEPS} \
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${lr} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_float16
}

# =============================================================================
# ZO-Adam (MeZO-Adam)
# =============================================================================
run_zo_adam() {
    local lr="1e-5"
    echo "======================================"
    echo "[2/4] Running ZO-Adam (MeZO-Adam) with lr=${lr}"
    echo "======================================"

    export WANDB_PROJECT="mistral7b_memory_test"

    python run.py \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/zo_adam_${TASK} \
        --num_train_epochs=5 \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --load_best_model_at_end \
        --evaluation_strategy=steps \
        --save_strategy=steps \
        --save_total_limit=1 \
        --max_steps=${MAX_STEPS} \
        --logging_steps=10 \
        --num_eval=1000 \
        --num_train=${NUM_TRAIN} \
        --num_dev=${NUM_DEV} \
        --train_as_classification \
        --perturbation_mode=two_side \
        --trainer=zo_adam \
        --train_set_seed=0 \
        --lr_scheduler_type=constant \
        --eval_steps=${EVAL_STEPS} \
        --save_steps=${SAVE_STEPS} \
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${lr} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_float16
}

# =============================================================================
# HiZOO
# =============================================================================
run_hizoo() {
    local lr="1e-6"
    echo "======================================"
    echo "[3/4] Running HiZOO with lr=${lr}"
    echo "======================================"

    export WANDB_PROJECT="mistral7b_memory_test"

    python run.py \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/hizoo_${TASK} \
        --num_train_epochs=5 \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --load_best_model_at_end \
        --evaluation_strategy=steps \
        --save_strategy=steps \
        --save_total_limit=1 \
        --max_steps=${MAX_STEPS} \
        --logging_steps=10 \
        --num_eval=1000 \
        --num_train=${NUM_TRAIN} \
        --num_dev=${NUM_DEV} \
        --train_as_classification \
        --trainer=hizoo \
        --train_set_seed=0 \
        --lr_scheduler_type=constant \
        --eval_steps=${EVAL_STEPS} \
        --save_steps=${SAVE_STEPS} \
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${lr} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_float16
}

# =============================================================================
# BSZO-V4
# =============================================================================
run_bszo_v4() {
    local lr="1e-6"
    echo "======================================"
    echo "[4/4] Running BSZO-V4 with lr=${lr}"
    echo "======================================"

    export WANDB_PROJECT="mistral7b_memory_test"

    python run.py \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/bszo_v4_${TASK} \
        --num_train_epochs=5 \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --load_best_model_at_end \
        --evaluation_strategy=steps \
        --save_strategy=steps \
        --save_total_limit=1 \
        --max_steps=${MAX_STEPS} \
        --logging_steps=10 \
        --num_eval=1000 \
        --num_train=${NUM_TRAIN} \
        --num_dev=${NUM_DEV} \
        --train_as_classification \
        --trainer=bszo_v4 \
        --train_set_seed=0 \
        --lr_scheduler_type=constant \
        --eval_steps=${EVAL_STEPS} \
        --save_steps=${SAVE_STEPS} \
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${lr} \
        --zo_eps=${ZO_EPS} \
        --bszo_fixed_subspace_dim=2 \
        --bayesian_num_samples=3 \
        --bayesian_adaptive_sampling=True \
        --bayesian_one_sided=True \
        --load_float16
}

# =============================================================================
# Main: Run all methods sequentially
# =============================================================================

# Parse argument (optional: run specific method)
METHOD=${1:-"all"}

case $METHOD in
    zo_sgd)
        run_zo_sgd
        ;;
    zo_adam)
        run_zo_adam
        ;;
    hizoo)
        run_hizoo
        ;;
    bszo_v4)
        run_bszo_v4
        ;;
    all)
        run_zo_sgd
        run_zo_adam
        run_hizoo
        run_bszo_v4
        ;;
    *)
        echo "Usage: $0 [zo_sgd|zo_adam|hizoo|bszo_v4|all]"
        exit 1
        ;;
esac

echo "============================================="
echo "Memory test completed!"
echo "Check wandb for GPU memory usage"
echo "============================================="

#!/bin/bash
# =============================================================================
# HiZOO experiments for Mistral-7B
# Datasets: WSC, Copa, TREC, WIC
# Note: HiZOO requires 2 GPUs for each task
#
# Usage:
#   Single dataset:  ./run_hizoo_mistral7b.sh WSC
#   All sequential:  ./run_hizoo_mistral7b.sh all
#   Custom GPUs:     CUDA_VISIBLE_DEVICES=0,1 ./run_hizoo_mistral7b.sh WSC
#
# Examples:
#   ./run_hizoo_mistral7b.sh WSC              # WSC on GPU 2,3 (default)
#   ./run_hizoo_mistral7b.sh Copa             # Copa on GPU 2,3 (default)
#   ./run_hizoo_mistral7b.sh all              # All datasets sequentially
#   CUDA_VISIBLE_DEVICES=0,1 ./run_hizoo_mistral7b.sh WSC
# =============================================================================

MODEL_NAME="mistralai/Mistral-7B-v0.1"
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
EARLY_STOPPING=8
ZO_EPS="1e-4"
LRS=("1e-6" "5e-7" "1e-7" "5e-8" "1e-8")

# Default to GPU 2,3 if not specified
GPUS=${CUDA_VISIBLE_DEVICES:-"2,3"}

TASK=${1:-"all"}
DATASETS=("WSC" "Copa" "TREC" "WIC")

# Dataset-specific parameters (train_size, num_train, num_dev)
# WSC: train=554 -> num_train=450, num_dev=100
# Copa: train=400 -> num_train=300, num_dev=100
# TREC: train=5500 -> num_train=1000, num_dev=500
# WIC: train=5428 -> num_train=1000, num_dev=500
get_data_params() {
    local task=$1
    case $task in
        WSC)     echo "450 100" ;;   # num_train num_dev
        Copa)    echo "300 100" ;;
        TREC)    echo "1000 500" ;;
        WIC)     echo "1000 500" ;;
        *)       echo "1000 500" ;;
    esac
}

run_task() {
    local TASK=$1
    export WANDB_PROJECT="hizoo_${TASK,,}_mistral7b_cls_ft"

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "GPUs ${GPUS} | HiZOO | ${TASK} | lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        CUDA_VISIBLE_DEVICES=$GPUS python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/hizoo_mistral7b_${TASK}_lr${lr} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --eval_strategy=steps \
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
            --overwrite_output_dir \
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${lr} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0 \
            --load_float16
    done
}

# Main logic
echo "Using GPUs: ${GPUS}"

if [ "$TASK" == "all" ]; then
    for dataset in "${DATASETS[@]}"; do
        run_task $dataset
    done
else
    run_task $TASK
fi

echo "HiZOO Mistral-7B experiments completed!"

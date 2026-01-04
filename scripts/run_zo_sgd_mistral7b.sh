#!/bin/bash
# =============================================================================
# ZO-SGD experiments for Mistral-7B
# Datasets: WSC, Copa, TREC, WIC
#
# Usage:
#   Single dataset:  ./run_zo_sgd_mistral7b.sh WSC 0
#   All sequential:  ./run_zo_sgd_mistral7b.sh all 0
#   Two GPUs parallel: CUDA_VISIBLE_DEVICES=2,3 ./run_zo_sgd_mistral7b.sh all_parallel
#
# Examples:
#   ./run_zo_sgd_mistral7b.sh WSC 0           # WSC on GPU 0
#   ./run_zo_sgd_mistral7b.sh Copa 1          # Copa on GPU 1
#   CUDA_VISIBLE_DEVICES=0,1 ./run_zo_sgd_mistral7b.sh all_parallel
#   CUDA_VISIBLE_DEVICES=2,3 ./run_zo_sgd_mistral7b.sh all_parallel
# =============================================================================

MODEL_NAME="mistralai/Mistral-7B-v0.1"
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
EARLY_STOPPING=8
ZO_EPS="1e-4"
LRS=("5e-7" "1e-7" "5e-8" "1e-8" "5e-9")

TASK=${1:-"all"}
GPU_ID=${2:-2}
DATASETS=("TREC" "WIC")

# Dataset-specific parameters (train_size, num_train, num_dev)
# WSC: train=554 → num_train=450, num_dev=100 → actual_train=450
# Copa: train=400 → num_train=300, num_dev=100 → actual_train=300
# TREC: train=5500 → num_train=1000, num_dev=500 → actual_train=500
# WIC: train=5428 → num_train=1000, num_dev=500 → actual_train=500
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
    local GPU=$2
    export WANDB_PROJECT="zo_sgd_${TASK,,}_mistral7b_ft"

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "GPU ${GPU} | ZO-SGD | ${TASK} | lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        CUDA_VISIBLE_DEVICES=$GPU python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/zo_sgd_mistral7b_${TASK}_lr${lr} \
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

run_parallel() {
    # Parse CUDA_VISIBLE_DEVICES to get the two GPU IDs
    # e.g., CUDA_VISIBLE_DEVICES=2,3 -> GPU0=2, GPU1=3
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "Warning: CUDA_VISIBLE_DEVICES not set, using 0,1"
        GPU0=0
        GPU1=1
    else
        IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
        GPU0=${GPU_ARRAY[0]}
        GPU1=${GPU_ARRAY[1]:-$GPU0}
    fi

    echo "Starting parallel runs: GPU ${GPU0} and GPU ${GPU1}"
    echo "GPU ${GPU0}: TREC"
    echo "GPU ${GPU1}: WIC"

    # GPU0 runs TREC
    (
        run_task "TREC" $GPU0
    ) &
    PID_GPU0=$!

    # GPU1 runs WIC
    (
        run_task "WIC" $GPU1
    ) &
    PID_GPU1=$!

    # Wait for both to complete
    wait $PID_GPU0
    wait $PID_GPU1

    echo "All parallel experiments completed!"
}

# Main logic
if [ "$TASK" == "all_parallel" ]; then
    run_parallel
elif [ "$TASK" == "all" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    echo "Using GPU: ${GPU_ID}"
    for dataset in "${DATASETS[@]}"; do
        run_task $dataset $GPU_ID
    done
else
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    echo "Using GPU: ${GPU_ID}"
    run_task $TASK $GPU_ID
fi

echo "ZO-SGD Mistral-7B experiments completed!"

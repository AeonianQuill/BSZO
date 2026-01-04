#!/bin/bash
# =============================================================================
# RoBERTa-Large ZO Fine-tuning Experiments
# Datasets: SST2, RTE, CB, WIC, TREC
# Methods: zo_sgd, zo_adam, hizoo, bszo_v4
#
# Usage:
#   bash run_roberta_large_all.sh [METHOD] [DATASET] [GPU_ID]
#
# Examples:
#   bash run_roberta_large_all.sh all all 0        # All on GPU 0
#   bash run_roberta_large_all.sh zo_sgd all 0     # zo_sgd all datasets on GPU 0
#   bash run_roberta_large_all.sh all SST2 1       # All methods on SST2 on GPU 1
#
# Two GPUs parallel example (run in two terminals):
#   Terminal 1: bash run_roberta_large_all.sh zo_sgd,zo_adam all 0
#   Terminal 2: bash run_roberta_large_all.sh hizoo,bszo_v4 all 1
# =============================================================================

# Common settings
MODEL_NAME="roberta-large"
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
ZO_EPS="1e-4"

# Datasets (SST2, RTE, WIC, TREC already completed, only CB remaining)
DATASETS=("CB")

# Dataset-specific parameters (num_train, num_dev)
# CB: train=250 → num_train=200, num_dev=50 → actual_train=200
get_data_params() {
    local task=$1
    case $task in
        CB)   echo "200 50" ;;
        *)    echo "1000 500" ;;
    esac
}

# Function to get num_eval for each dataset
get_num_eval() {
    local task=$1
    case $task in
        SST2) echo 1000 ;;
        RTE) echo 277 ;;
        CB) echo 56 ;;
        WIC) echo 1000 ;;
        TREC) echo 500 ;;
        *) echo 1000 ;;
    esac
}

# =============================================================================
# ZO-SGD
# =============================================================================
run_zo_sgd() {
    local TASK=$1
    local LRS=("1e-4" "1e-5" "1e-6")
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running ZO-SGD on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="zo_sgd_${TASK_LOWER}_roberta_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/zo_sgd_${TASK}_lr${lr} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --evaluation_strategy=steps \
            --save_strategy=steps \
            --save_total_limit=1 \
            --max_steps=${MAX_STEPS} \
            --logging_steps=10 \
            --num_eval=$(get_num_eval $TASK) \
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
            --use_roberta_mlm
    done
}

# =============================================================================
# ZO-Adam
# =============================================================================
run_zo_adam() {
    local TASK=$1
    local LRS=("1e-4" "1e-5" "1e-6")
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running ZO-Adam on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="zo_adam_${TASK_LOWER}_roberta_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/zo_adam_${TASK}_lr${lr} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --evaluation_strategy=steps \
            --save_strategy=steps \
            --save_total_limit=1 \
            --max_steps=${MAX_STEPS} \
            --logging_steps=10 \
            --num_eval=$(get_num_eval $TASK) \
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
            --use_roberta_mlm
    done
}

# =============================================================================
# HiZOO
# =============================================================================
run_hizoo() {
    local TASK=$1
    local LRS=("1e-4" "1e-5" "1e-6")
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running HiZOO on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="hizoo_${TASK_LOWER}_roberta_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/hizoo_${TASK}_lr${lr} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --evaluation_strategy=steps \
            --save_strategy=steps \
            --save_total_limit=1 \
            --max_steps=${MAX_STEPS} \
            --logging_steps=10 \
            --num_eval=$(get_num_eval $TASK) \
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
            --use_roberta_mlm
    done
}

# =============================================================================
# BSZO-V4
# =============================================================================
run_bszo_v4() {
    local TASK=$1
    local LRS=("1e-4" "1e-5" "1e-6")
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    # TREC uses different project naming
    local PROJECT_PREFIX="v4"
    if [ "$TASK" == "TREC" ]; then
        PROJECT_PREFIX="bszo_v4"
    fi

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running BSZO-V4 on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="${PROJECT_PREFIX}_${TASK_LOWER}_roberta_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/bszo_v4_${TASK}_lr${lr} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --evaluation_strategy=steps \
            --save_strategy=steps \
            --save_total_limit=1 \
            --max_steps=${MAX_STEPS} \
            --logging_steps=10 \
            --num_eval=$(get_num_eval $TASK) \
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
            --weight_decay=0 \
            --use_roberta_mlm
    done
}

# =============================================================================
# Main: Run all experiments
# =============================================================================
echo "Starting RoBERTa-Large ZO Fine-tuning Experiments"
echo "=================================================="
echo "Datasets: ${DATASETS[*]}"
echo "Methods: zo_sgd, zo_adam, hizoo, bszo_v4"
echo "=================================================="

# Parse arguments
METHOD=${1:-"all"}
DATASET=${2:-"all"}
GPU_ID=${3:-"0"}

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "Using GPU: ${GPU_ID}"

run_single_method() {
    local method=$1
    local dataset=$2

    case $method in
        zo_sgd)
            run_zo_sgd $dataset
            ;;
        zo_adam)
            run_zo_adam $dataset
            ;;
        hizoo)
            run_hizoo $dataset
            ;;
        bszo_v4)
            run_bszo_v4 $dataset
            ;;
    esac
}

# Support comma-separated methods (e.g., "zo_sgd,zo_adam")
# Save and restore IFS to avoid side effects
OLD_IFS="$IFS"
IFS=',' read -ra METHOD_LIST <<< "$METHOD"
IFS=',' read -ra DATASET_LIST <<< "$DATASET"
IFS="$OLD_IFS"

# If "all" is specified, expand to full list
if [ "$METHOD" == "all" ]; then
    METHOD_LIST=("zo_sgd" "zo_adam" "hizoo" "bszo_v4")
fi
if [ "$DATASET" == "all" ]; then
    DATASET_LIST=("CB")
fi

echo "Methods to run: ${METHOD_LIST[*]}"
echo "Datasets to run: ${DATASET_LIST[*]}"

# Run all combinations of specified methods and datasets
for dataset in "${DATASET_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        echo ">>> Running: method=$method, dataset=$dataset"
        run_single_method "$method" "$dataset"
    done
done

echo "=================================================="
echo "All experiments completed!"
echo "=================================================="

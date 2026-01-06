#!/bin/bash
# =============================================================================
# RoBERTa-Large ZO Fine-tuning Experiments with Best Learning Rates
# Datasets: SST2, BoolQ, RTE, CB, WIC, TREC
# Methods: zo_sgd (MeZO), zo_adam (MeZO-Adam), hizoo, bszo_v3, bszo_v4
#
# Each configuration runs twice (run 1, run 2) with train_set_seed=0 fixed
# seed is NOT set, allowing natural model randomness for variance estimation
#
# Usage:
#   bash run_roberta_large_best_lr.sh [METHOD] [DATASET] [GPU_ID]
#
# Examples:
#   bash run_roberta_large_best_lr.sh all all 0        # All on GPU 0
#   bash run_roberta_large_best_lr.sh zo_sgd SST2 0    # zo_sgd on SST2 on GPU 0
# =============================================================================

# Wandb offline mode (sync later)
export WANDB_MODE=offline
export WANDB_DIR=/mnt/innovator/chl/wandb_cache
export WANDB_CACHE_DIR=/mnt/innovator/chl/wandb_cache

# Set cache directories
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1

# Common settings
# Use local path if downloaded from ModelScope, otherwise use HuggingFace name
MODEL_NAME="/mnt/innovator/chl/models/roberta-large"
# MODEL_NAME="roberta-large"  # Use this if loading from HuggingFace directly
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
ZO_EPS="1e-4"

# All datasets to run
DATASETS=("SST2" "RTE" "CB" "WIC" "TREC")

# Dataset-specific parameters (num_train, num_dev)
get_data_params() {
    local task=$1
    case $task in
        SST2)   echo "1000 500" ;;
        RTE)    echo "1000 500" ;;
        CB)     echo "200 50" ;;
        WIC)    echo "1000 500" ;;
        TREC)   echo "1000 500" ;;
        *)      echo "1000 500" ;;
    esac
}

# Function to get num_eval for each dataset
get_num_eval() {
    local task=$1
    case $task in
        SST2)   echo 1000 ;;
        RTE)    echo 277 ;;
        CB)     echo 56 ;;
        WIC)    echo 1000 ;;
        TREC)   echo 500 ;;
        *)      echo 1000 ;;
    esac
}

# =============================================================================
# Learning rates to run (excluding already completed ones)
# Full grid: [1e-4, 1e-5, 1e-6, 1e-7]
# Each method x dataset has one LR already done, run the other 3
# =============================================================================

# ZO-SGD: already ran 1e-6 for all tasks
get_lrs_zo_sgd() {
    local task=$1
    # All tasks already ran 1e-6
    echo "1e-4 1e-5 1e-7"
}

# ZO-Adam: SST2/RTE/WIC/TREC=1e-5, CB=1e-4
get_lrs_zo_adam() {
    local task=$1
    case $task in
        CB)     echo "1e-5 1e-6 1e-7" ;;  # exclude 1e-4
        *)      echo "1e-4 1e-6 1e-7" ;;  # exclude 1e-5
    esac
}

# HiZOO: already ran 1e-6 for all tasks
get_lrs_hizoo() {
    local task=$1
    # All tasks already ran 1e-6
    echo "1e-4 1e-6 1e-5 1e-7"
}

# V3: SST2/RTE/CB/WIC=1e-6, TREC=1e-5
get_lrs_v3() {
    local task=$1
    case $task in
        TREC)   echo "1e-4 1e-6 1e-7" ;;  # exclude 1e-5
        *)      echo "1e-4 1e-5 1e-7" ;;  # exclude 1e-6
    esac
}

# V4: SST2/RTE/CB/WIC=1e-6, TREC=1e-5
get_lrs_v4() {
    local task=$1
    case $task in
        TREC)   echo "1e-4 1e-6 1e-7" ;;  # exclude 1e-5
        *)      echo "1e-4 1e-5 1e-7" ;;  # exclude 1e-6
    esac
}

# =============================================================================
# ZO-SGD (MeZO)
# =============================================================================
run_zo_sgd() {
    local TASK=$1
    local LRS=$(get_lrs_zo_sgd $TASK)  # Pass TASK to get remaining LRs
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for LR in ${LRS}; do
        echo "======================================"
        echo "Running ZO-SGD on ${TASK} with lr=${LR}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="roberta_grid_${TASK_LOWER}"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/roberta_zo_sgd_${TASK}_lr${LR} \
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
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0 \
            --use_roberta_mlm

        echo "[ZO-SGD] ${TASK} lr=${LR} completed!"
    done
}

# =============================================================================
# ZO-Adam (MeZO-Adam)
# =============================================================================
run_zo_adam() {
    local TASK=$1
    local LRS=$(get_lrs_zo_adam $TASK)
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for LR in ${LRS}; do
        echo "======================================"
        echo "Running ZO-Adam on ${TASK} with lr=${LR}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="roberta_grid_${TASK_LOWER}"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/roberta_zo_adam_${TASK}_lr${LR} \
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
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0 \
            --use_roberta_mlm

        echo "[ZO-Adam] ${TASK} lr=${LR} completed!"
    done
}

# =============================================================================
# HiZOO
# =============================================================================
run_hizoo() {
    local TASK=$1
    local LRS=$(get_lrs_hizoo $TASK)
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for LR in ${LRS}; do
        echo "======================================"
        echo "Running HiZOO on ${TASK} with lr=${LR}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="roberta_grid_${TASK_LOWER}"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/roberta_hizoo_${TASK}_lr${LR} \
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
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0 \
            --use_roberta_mlm

        echo "[HiZOO] ${TASK} lr=${LR} completed!"
    done
}

# =============================================================================
# BSZO-V3
# =============================================================================
run_bszo_v3() {
    local TASK=$1
    local LRS=$(get_lrs_v3 $TASK)  # Pass TASK to get remaining LRs
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for LR in ${LRS}; do
        echo "======================================"
        echo "Running BSZO-V3 on ${TASK} with lr=${LR}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="roberta_grid_${TASK_LOWER}"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/roberta_v3_${TASK}_lr${LR} \
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
            --trainer=bszo_v3 \
            --train_set_seed=0 \
            --lr_scheduler_type=constant \
            --eval_steps=${EVAL_STEPS} \
            --save_steps=${SAVE_STEPS} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --bszo_fixed_subspace_dim=2 \
            --bayesian_num_samples=3 \
            --bayesian_adaptive_sampling=True \
            --bayesian_one_sided=True \
            --weight_decay=0 \
            --use_roberta_mlm

        echo "[BSZO-V3] ${TASK} lr=${LR} completed!"
    done
}

# =============================================================================
# BSZO-V4
# =============================================================================
run_bszo_v4() {
    local TASK=$1
    local LRS=$(get_lrs_v4 $TASK)  # Pass TASK to get remaining LRs
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for LR in ${LRS}; do
        echo "======================================"
        echo "Running BSZO-V4 on ${TASK} with lr=${LR}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="roberta_grid_${TASK_LOWER}"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/roberta_v4_${TASK}_lr${LR} \
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
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --bszo_fixed_subspace_dim=2 \
            --bayesian_num_samples=3 \
            --bayesian_adaptive_sampling=True \
            --bayesian_one_sided=True \
            --weight_decay=0 \
            --use_roberta_mlm

        echo "[BSZO-V4] ${TASK} lr=${LR} completed!"
    done
}

# =============================================================================
# Main: Run all experiments
# =============================================================================
echo "=========================================================="
echo "RoBERTa-Large ZO Fine-tuning with Best Learning Rates"
echo "=========================================================="
echo "Datasets: ${DATASETS[*]}"
echo "Methods: zo_sgd (MeZO), zo_adam (MeZO-Adam), hizoo, v3, v4"
echo "Runs: ${RUN_IDS[*]} (train_set_seed=0 fixed, seed not set)"
echo "=========================================================="

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
        bszo_v3|v3)
            run_bszo_v3 $dataset
            ;;
        bszo_v4|v4)
            run_bszo_v4 $dataset
            ;;
    esac
}

# Support comma-separated methods and datasets
OLD_IFS="$IFS"
IFS=',' read -ra METHOD_LIST <<< "$METHOD"
IFS=',' read -ra DATASET_LIST <<< "$DATASET"
IFS="$OLD_IFS"

# If "all" is specified, expand to full list
if [ "$METHOD" == "all" ]; then
    METHOD_LIST=("zo_sgd" "zo_adam" "hizoo" "bszo_v3" "bszo_v4")
fi
if [ "$DATASET" == "all" ]; then
    DATASET_LIST=("SST2" "BoolQ" "RTE" "CB" "WIC" "TREC")
fi

echo "Methods to run: ${METHOD_LIST[*]}"
echo "Datasets to run: ${DATASET_LIST[*]}"
echo ""

# Print learning rate summary
echo "Learning Rate Summary:"
echo "------------------------------------------------------------------------"
printf "%-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" "Dataset" "MeZO" "Adam" "HiZOO" "V3" "V4"
echo "------------------------------------------------------------------------"
for ds in "${DATASET_LIST[@]}"; do
    printf "%-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
        "$ds" \
        "$(get_best_lr_zo_sgd $ds)" \
        "$(get_best_lr_zo_adam $ds)" \
        "$(get_best_lr_hizoo $ds)" \
        "$(get_best_lr_v3 $ds)" \
        "$(get_best_lr_v4 $ds)"
done
echo "------------------------------------------------------------------------"
echo ""

# Run all combinations of specified methods and datasets
for dataset in "${DATASET_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        echo ">>> Running: method=$method, dataset=$dataset"
        run_single_method "$method" "$dataset"
    done
done

echo "=========================================================="
echo "All experiments completed!"
echo "Total runs: ${#METHOD_LIST[@]} methods x ${#DATASET_LIST[@]} datasets x 2 runs"
echo "=========================================================="

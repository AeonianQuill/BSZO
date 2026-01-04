#!/bin/bash
# =============================================================================
# OPT-1.3B ZO Fine-tuning Experiments
# Datasets: TREC, WSC
# Methods: zo_sgd (mezo), zo_adam (mezo-adam), hizoo, bszo_v4 (v4)
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
MODEL_NAME="/mnt/innovator/chl/models/opt-1.3b"
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
EARLY_STOPPING=8
ZO_EPS="1e-4"

# Datasets
DATASETS=("TREC" "WSC")

# Dataset-specific parameters (num_train, num_dev)
# WSC: train=554 → num_train=450, num_dev=100 → actual_train=450
# TREC: train=5500 → num_train=1000, num_dev=500 → actual_train=500
get_data_params() {
    local task=$1
    case $task in
        WSC)     echo "450 100" ;;
        TREC)    echo "1000 500" ;;
        *)       echo "1000 500" ;;
    esac
}

# =============================================================================
# ZO-SGD (MeZO)
# LR range: [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
# =============================================================================
run_zo_sgd() {
    local TASK=$1
    local LRS=("1e-6" "5e-7" "1e-7" "5e-8" "1e-8")

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running ZO-SGD (MeZO) on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="zo_sgd_${TASK,,}_opt1.3b_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/opt1.3b_zo_sgd_${TASK}_lr${lr} \
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
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${lr} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0
    done
}

# =============================================================================
# ZO-Adam (MeZO-Adam)
# LR range: [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
# =============================================================================
run_zo_adam() {
    local TASK=$1
    local LRS=("1e-4" "5e-5" "1e-5" "5e-6" "1e-6")

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running ZO-Adam (MeZO-Adam) on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="zo_adam_${TASK,,}_opt1.3b_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/opt1.3b_zo_adam_${TASK}_lr${lr} \
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
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${lr} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0
    done
}

# =============================================================================
# HiZOO
# LR range: [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
# =============================================================================
run_hizoo() {
    local TASK=$1
    local LRS=("1e-5" "5e-6" "1e-6" "5e-7" "1e-7")

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running HiZOO on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="hizoo_${TASK,,}_opt1.3b_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/opt1.3b_hizoo_${TASK}_lr${lr} \
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
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${lr} \
            --zo_eps=${ZO_EPS} \
            --weight_decay=0
    done
}

# =============================================================================
# BSZO-V4
# LR range: [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
# =============================================================================
run_bszo_v4() {
    local TASK=$1
    local LRS=("1e-5" "5e-6" "1e-6" "5e-7" "1e-7")

    # Get dataset-specific parameters
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "Running BSZO-V4 on ${TASK} with lr=${lr}"
        echo "NUM_TRAIN=${NUM_TRAIN}, NUM_DEV=${NUM_DEV}"
        echo "======================================"

        export WANDB_PROJECT="bszo_v4_${TASK,,}_opt1.3b_ft"

        python run.py \
            --model_name=${MODEL_NAME} \
            --task_name=${TASK} \
            --output_dir=result/opt1.3b_bszo_v4_${TASK}_lr${lr} \
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
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${lr} \
            --zo_eps=${ZO_EPS} \
            --bszo_fixed_subspace_dim=2 \
            --bayesian_num_samples=3 \
            --bayesian_adaptive_sampling=True \
            --bayesian_one_sided=True
    done
}

# =============================================================================
# Main: Run all experiments
# =============================================================================
echo "Starting OPT-1.3B ZO Fine-tuning Experiments"
echo "============================================="
echo "Datasets: ${DATASETS[*]}"
echo "Methods: zo_sgd (mezo), zo_adam (mezo-adam), hizoo, bszo_v4 (v4)"
echo "============================================="

# Parse arguments
METHOD=${1:-"all"}
DATASET=${2:-"all"}

run_single_method() {
    local method=$1
    local dataset=$2

    case $method in
        zo_sgd|mezo)
            run_zo_sgd $dataset
            ;;
        zo_adam|mezo_adam|mezo-adam)
            run_zo_adam $dataset
            ;;
        hizoo)
            run_hizoo $dataset
            ;;
        bszo_v4|v4)
            run_bszo_v4 $dataset
            ;;
    esac
}

if [ "$METHOD" == "all" ] && [ "$DATASET" == "all" ]; then
    # Run all combinations
    for dataset in "${DATASETS[@]}"; do
        for method in zo_sgd zo_adam hizoo bszo_v4; do
            run_single_method $method $dataset
        done
    done
elif [ "$METHOD" == "all" ]; then
    # Run all methods for specific dataset
    for method in zo_sgd zo_adam hizoo bszo_v4; do
        run_single_method $method $DATASET
    done
elif [ "$DATASET" == "all" ]; then
    # Run specific method for all datasets
    for dataset in "${DATASETS[@]}"; do
        run_single_method $METHOD $dataset
    done
else
    # Run specific method and dataset
    run_single_method $METHOD $DATASET
fi

echo "============================================="
echo "All experiments completed!"
echo "Run 'wandb sync ./wandb/' to upload results"
echo "============================================="

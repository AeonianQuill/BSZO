#!/bin/bash
# =============================================================================
# BSZO V3 Main Experiments
# Models: Mistral-7B (fp16), OPT-13B (bf16)
# Datasets: SST2, RTE, WSC, Copa, TREC, WIC
# Total: 12 experiments (2 models x 6 datasets)
# =============================================================================

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR=/mnt/innovator/chl/wandb_cache
export WANDB_CACHE_DIR=/mnt/innovator/chl/wandb_cache

# Set cache directories
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1

# Use GPU from environment, default to GPU 0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# =============================================================================
# Common settings
# =============================================================================
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
EARLY_STOPPING=8
ZO_EPS="1e-4"
NUM_EVAL=1000

# Model paths
MODEL_MISTRAL_7B="/mnt/innovator/chl/models/mistral-7b"
MODEL_OPT_13B="/mnt/innovator/chl/models/opt-13b"

# =============================================================================
# Dataset-specific parameters
# =============================================================================
get_data_params() {
    local task=$1
    case $task in
        WSC)     echo "450 100" ;;
        Copa)    echo "300 100" ;;
        TREC)    echo "1000 500" ;;
        WIC)     echo "1000 500" ;;
        SST2)    echo "1000 500" ;;
        RTE)     echo "1000 500" ;;
        *)       echo "1000 500" ;;
    esac
}

# Mistral-7B learning rates (from v4 experiments)
get_mistral_lr() {
    local task=$1
    case $task in
        SST2)    echo "5e-8" ;;
        RTE)     echo "1e-7" ;;
        WSC)     echo "5e-8" ;;  # Multiple LRs for WSC
        Copa)    echo "1e-7" ;;
        TREC)    echo "1e-7" ;;
        WIC)     echo "1e-7" ;;
        *)       echo "1e-7" ;;
    esac
}

# OPT-13B learning rates (from v4 experiments)
get_opt13b_lr() {
    local task=$1
    case $task in
        SST2)    echo "1e-7" ;;
        RTE)     echo "5e-8" ;;
        WSC)     echo "1e-6 5e-6 1e-7 5e-8" ;;  # Multiple LRs for WSC
        Copa)    echo "5e-7" ;;
        TREC)    echo "1e-7" ;;
        WIC)     echo "1e-7" ;;
        *)       echo "1e-7" ;;
    esac
}

# =============================================================================
# Mistral-7B Experiments (fp16)
# =============================================================================
run_mistral_7b() {
    local TASK=$1
    local LRS=$(get_mistral_lr $TASK)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="bszo_v3_mistral7b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[Mistral-7B] BSZO-V3 on ${TASK}"
        echo "  lr=${LR}"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_MISTRAL_7B} \
            --task_name=${TASK} \
            --output_dir=result/v3_mistral7b_${TASK}_lr${LR} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --evaluation_strategy=steps \
            --save_strategy=steps \
            --save_total_limit=1 \
            --max_steps=${MAX_STEPS} \
            --logging_steps=10 \
            --num_eval=${NUM_EVAL} \
            --num_train=${NUM_TRAIN} \
            --num_dev=${NUM_DEV} \
            --train_as_classification \
            --trainer=bszo_v3 \
            --train_set_seed=0 \
            --lr_scheduler_type=constant \
            --eval_steps=${EVAL_STEPS} \
            --save_steps=${SAVE_STEPS} \
            --overwrite_output_dir \
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --bszo_fixed_subspace_dim=2 \
            --bayesian_num_samples=3 \
            --bayesian_adaptive_sampling=True \
            --bayesian_one_sided=True \
            --weight_decay=0 \
            --load_float16

        echo "[Mistral-7B] ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# OPT-13B Experiments (bf16)
# =============================================================================
run_opt_13b() {
    local TASK=$1
    local LRS=$(get_opt13b_lr $TASK)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="bszo_v3_opt13b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[OPT-13B] BSZO-V3 on ${TASK}"
        echo "  lr=${LR}"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_OPT_13B} \
            --task_name=${TASK} \
            --output_dir=result/v3_opt13b_${TASK}_lr${LR} \
            --num_train_epochs=5 \
            --per_device_train_batch_size=${BATCH_SIZE} \
            --load_best_model_at_end \
            --evaluation_strategy=steps \
            --save_strategy=steps \
            --save_total_limit=1 \
            --max_steps=${MAX_STEPS} \
            --logging_steps=10 \
            --num_eval=${NUM_EVAL} \
            --num_train=${NUM_TRAIN} \
            --num_dev=${NUM_DEV} \
            --train_as_classification \
            --trainer=bszo_v3 \
            --train_set_seed=0 \
            --lr_scheduler_type=constant \
            --eval_steps=${EVAL_STEPS} \
            --save_steps=${SAVE_STEPS} \
            --overwrite_output_dir \
            --early_stopping_patience=${EARLY_STOPPING} \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --bszo_fixed_subspace_dim=2 \
            --bayesian_num_samples=3 \
            --bayesian_adaptive_sampling=True \
            --bayesian_one_sided=True \
            --weight_decay=0 \
            --load_bfloat16

        echo "[OPT-13B] ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# Main
# =============================================================================
echo "============================================="
echo "BSZO V3 Main Experiments"
echo "============================================="
echo "Models: Mistral-7B (fp16), OPT-13B (bf16)"
echo "Datasets: SST2, RTE, WSC, Copa, TREC, WIC"
echo "Total: 12 experiments"
echo "============================================="
echo ""

# Usage: ./run_v3_main_experiments.sh [model] [task]
# model: mistral7b | opt13b | all
# task: SST2 | RTE | WSC | Copa | TREC | WIC | all

MODEL=${1:-"all"}
TASK=${2:-"all"}

DATASETS=("SST2" "RTE" "WSC" "Copa" "TREC" "WIC")

if [ "$MODEL" == "mistral7b" ]; then
    if [ "$TASK" == "all" ]; then
        for task in "${DATASETS[@]}"; do
            run_mistral_7b $task
        done
    else
        run_mistral_7b $TASK
    fi
elif [ "$MODEL" == "opt13b" ]; then
    if [ "$TASK" == "all" ]; then
        for task in "${DATASETS[@]}"; do
            run_opt_13b $task
        done
    else
        run_opt_13b $TASK
    fi
elif [ "$MODEL" == "all" ]; then
    if [ "$TASK" == "all" ]; then
        # Run all 12 experiments
        echo "Running all 12 experiments..."
        echo ""

        echo "=== Mistral-7B (6 experiments) ==="
        for task in "${DATASETS[@]}"; do
            run_mistral_7b $task
        done

        echo "=== OPT-13B (6 experiments) ==="
        for task in "${DATASETS[@]}"; do
            run_opt_13b $task
        done
    else
        # Run specific task on both models
        run_mistral_7b $TASK
        run_opt_13b $TASK
    fi
else
    echo "============================================="
    echo "Usage: $0 [model] [task]"
    echo ""
    echo "Models: mistral7b | opt13b | all"
    echo "Tasks:  SST2 | RTE | WSC | Copa | TREC | WIC | all"
    echo ""
    echo "Examples:"
    echo "  $0 all all          # Run all 12 experiments"
    echo "  $0 mistral7b all    # Run Mistral-7B on all datasets"
    echo "  $0 opt13b SST2      # Run OPT-13B on SST2"
    echo "  $0 all WSC          # Run both models on WSC"
    echo "============================================="
    exit 1
fi

echo "============================================="
echo "All experiments completed!"
echo "============================================="
echo ""
echo "Learning rates used:"
echo ""
echo "Mistral-7B (fp16):"
echo "  SST2: 5e-8, RTE: 1e-7, WSC: 1e-6/5e-6/5e-7/1e-7"
echo "  Copa: 1e-7, TREC: 1e-7, WIC: 1e-7"
echo ""
echo "OPT-13B (bf16):"
echo "  SST2: 1e-7, RTE: 1e-7, WSC: 5e-7/5e-6/1e-7"
echo "  Copa: 5e-7, TREC: 1e-7, WIC: 1e-7"
echo "============================================="

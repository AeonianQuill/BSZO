#!/bin/bash
# =============================================================================
# HiZOO Main Experiments (Re-run with corrected hizoo_smooth=1e-10)
# Models: OPT-1.3B, Mistral-7B (fp16), OPT-13B (bf16)
# Datasets: SST2, RTE, WSC, Copa, TREC, WIC
#
# Usage:
#   bash run_hizoo_main_experiments.sh [model] [task] [gpu_id]
#   model: opt1.3b | mistral7b | opt13b | all
#   task: SST2 | RTE | WSC | Copa | TREC | WIC | all
#   gpu_id: 0, 1, 2, ...
#
# Examples:
#   bash run_hizoo_main_experiments.sh opt1.3b all 0
#   bash run_hizoo_main_experiments.sh mistral7b SST2 1
#   bash run_hizoo_main_experiments.sh all all 0
# =============================================================================

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR=/mnt/innovator/chl/wandb_cache
export WANDB_CACHE_DIR=/mnt/innovator/chl/wandb_cache

# Set cache directories
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Use GPU from argument, default to GPU 0
GPU_ID=${3:-"1"}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

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
MODEL_OPT_1_3B="/mnt/innovator/chl/models/opt-1.3b"
MODEL_MISTRAL_7B="/mnt/innovator/chl/models/mistral-7b"
MODEL_OPT_13B="/mnt/innovator/chl/models/opt-13b"

# Datasets
DATASETS=("SST2" "RTE" "WSC" "Copa" "TREC" "WIC")

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

# =============================================================================
# Learning rates (grid search)
# =============================================================================
# OPT-1.3B: [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
get_opt1_3b_lrs() {
    echo "1e-5 5e-6 1e-6 5e-7 1e-7"
}

# OPT-13B: [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
get_opt13b_lrs() {
    echo "1e-5 5e-6 1e-6 5e-7 1e-7"
}

# Mistral-7B: [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
get_mistral_lrs() {
    echo "1e-6 5e-7 1e-7 5e-8 1e-8"
}

# =============================================================================
# OPT-1.3B Experiments
# =============================================================================
run_opt_1_3b() {
    local TASK=$1
    local LRS=$(get_opt1_3b_lrs)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="hizoo_opt1.3b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[OPT-1.3B] HiZOO on ${TASK}"
        echo "  lr=${LR}"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  early_stopping=${EARLY_STOPPING}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_OPT_1_3B} \
            --task_name=${TASK} \
            --output_dir=result/hizoo_opt1.3b_${TASK}_lr${LR} \
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
            --trainer=hizoo \
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
            --weight_decay=0

        echo "[OPT-1.3B] HiZOO ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# Mistral-7B Experiments (fp16)
# =============================================================================
run_mistral_7b() {
    local TASK=$1
    local LRS=$(get_mistral_lrs)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="hizoo_mistral7b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[Mistral-7B] HiZOO on ${TASK} (fp16)"
        echo "  lr=${LR}"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  early_stopping=${EARLY_STOPPING}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_MISTRAL_7B} \
            --task_name=${TASK} \
            --output_dir=result/hizoo_mistral7b_${TASK}_lr${LR} \
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
            --trainer=hizoo \
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
            --weight_decay=0 \
            --load_float16

        echo "[Mistral-7B] HiZOO ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# OPT-13B Experiments (bf16)
# =============================================================================
run_opt_13b() {
    local TASK=$1
    local LRS=$(get_opt13b_lrs)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="hizoo_opt13b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[OPT-13B] HiZOO on ${TASK} (bf16)"
        echo "  lr=${LR}"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  early_stopping=${EARLY_STOPPING}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_OPT_13B} \
            --task_name=${TASK} \
            --output_dir=result/hizoo_opt13b_${TASK}_lr${LR} \
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
            --trainer=hizoo \
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
            --weight_decay=0 \
            --load_bfloat16

        echo "[OPT-13B] HiZOO ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# Main
# =============================================================================
echo "============================================="
echo "HiZOO Main Experiments (Re-run)"
echo "============================================="
echo "Models: OPT-1.3B, Mistral-7B (fp16), OPT-13B (bf16)"
echo "Datasets: ${DATASETS[*]}"
echo "Note: Using corrected hizoo_smooth=1e-10 for zo_eps=1e-4"
echo "============================================="
echo ""

MODEL=${1:-"all"}
TASK=${2:-"all"}

echo "Using GPU: ${GPU_ID}"
echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo ""

# Run based on model selection
case $MODEL in
    opt1.3b)
        if [ "$TASK" == "all" ]; then
            for task in "${DATASETS[@]}"; do
                run_opt_1_3b $task
            done
        else
            run_opt_1_3b $TASK
        fi
        ;;
    mistral7b)
        if [ "$TASK" == "all" ]; then
            for task in "${DATASETS[@]}"; do
                run_mistral_7b $task
            done
        else
            run_mistral_7b $TASK
        fi
        ;;
    opt13b)
        if [ "$TASK" == "all" ]; then
            for task in "${DATASETS[@]}"; do
                run_opt_13b $task
            done
        else
            run_opt_13b $TASK
        fi
        ;;
    all)
        echo "Running all models..."
        echo ""

        echo "=== OPT-1.3B (5 LRs x 6 datasets = 30 experiments) ==="
        for task in "${DATASETS[@]}"; do
            run_opt_1_3b $task
        done

        echo "=== Mistral-7B fp16 (5 LRs x 6 datasets = 30 experiments) ==="
        for task in "${DATASETS[@]}"; do
            run_mistral_7b $task
        done

        echo "=== OPT-13B bf16 (5 LRs x 6 datasets = 30 experiments) ==="
        for task in "${DATASETS[@]}"; do
            run_opt_13b $task
        done
        ;;
    *)
        echo "============================================="
        echo "Usage: $0 [model] [task] [gpu_id]"
        echo ""
        echo "Models: opt1.3b | mistral7b | opt13b | all"
        echo "Tasks:  SST2 | RTE | WSC | Copa | TREC | WIC | all"
        echo ""
        echo "Learning rates:"
        echo "  OPT-1.3B:   [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]"
        echo "  OPT-13B:    [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]"
        echo "  Mistral-7B: [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]"
        echo ""
        echo "Examples:"
        echo "  $0 opt1.3b all 0       # OPT-1.3B on all datasets, GPU 0"
        echo "  $0 mistral7b SST2 1    # Mistral-7B on SST2, GPU 1"
        echo "  $0 opt13b all 2        # OPT-13B on all datasets, GPU 2"
        echo "  $0 all all 0           # All models (90 experiments)"
        echo "============================================="
        exit 1
        ;;
esac

echo "============================================="
echo "HiZOO experiments completed!"
echo "============================================="
echo ""
echo "Summary:"
echo "  OPT-1.3B:   LRs=[1e-5, 5e-6, 1e-6, 5e-7, 1e-7]"
echo "  Mistral-7B: LRs=[1e-6, 5e-7, 1e-7, 5e-8, 1e-8] (fp16)"
echo "  OPT-13B:    LRs=[1e-5, 5e-6, 1e-6, 5e-7, 1e-7] (bf16)"
echo ""
echo "Total: 3 models x 6 datasets x 5 LRs = 90 experiments"
echo "============================================="

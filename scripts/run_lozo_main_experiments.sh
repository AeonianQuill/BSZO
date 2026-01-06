#!/bin/bash
# =============================================================================
# LOZO Main Experiments
# Models: RoBERTa-Large, OPT-1.3B, Mistral-7B (fp16), OPT-13B (bf16)
#
# RoBERTa-Large: Learning rate grid search [1e-4, 1e-5, 1e-6], NO early stopping
# Others: Early stopping enabled, learning rates TBD
#
# Usage:
#   bash run_lozo_main_experiments.sh [model] [task] [gpu_id]
#   model: roberta | opt1.3b | mistral7b | opt13b | all
#   task: SST2 | RTE | ... | all
#   gpu_id: 0, 1, 2, ...
#
# Examples:
#   bash run_lozo_main_experiments.sh roberta all 0
#   bash run_lozo_main_experiments.sh opt13b SST2 1
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
GPU_ID=${3:-"0"}
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

# LOZO specific parameters (matching official defaults)
LOZO_RANK=2
LOZO_STEP_INTERVAL=50

# Model paths
MODEL_ROBERTA="/mnt/innovator/chl/models/roberta-large"
MODEL_OPT_1_3B="/mnt/innovator/chl/models/opt-1.3b"
MODEL_MISTRAL_7B="/mnt/innovator/chl/models/mistral-7b"
MODEL_OPT_13B="/mnt/innovator/chl/models/opt-13b"

# =============================================================================
# Dataset configurations
# =============================================================================
# RoBERTa datasets (5 tasks)
ROBERTA_DATASETS=("SST2" "RTE" "CB" "WIC" "TREC")

# Large model datasets (6 tasks)
LLM_DATASETS=("SST2" "RTE" "WSC" "Copa" "TREC" "WIC")

# Dataset-specific parameters (num_train, num_dev)
get_data_params() {
    local task=$1
    case $task in
        SST2)   echo "1000 500" ;;
        BoolQ)  echo "1000 500" ;;
        RTE)    echo "1000 500" ;;
        CB)     echo "200 50" ;;
        WIC)    echo "1000 500" ;;
        TREC)   echo "1000 500" ;;
        WSC)    echo "450 100" ;;
        Copa)   echo "300 100" ;;
        *)      echo "1000 500" ;;
    esac
}

# num_eval for each dataset
get_num_eval() {
    local task=$1
    case $task in
        RTE)    echo 277 ;;
        CB)     echo 56 ;;
        TREC)   echo 500 ;;
        *)      echo 1000 ;;
    esac
}

# =============================================================================
# Learning rates (same as MeZO/ZO-SGD)
# =============================================================================
# RoBERTa-Large: Grid search [1e-4, 1e-5, 1e-6]
get_roberta_lrs() {
    echo "1e-4 1e-5 1e-6"
}

# OPT-1.3B: Grid search [1e-6, 5e-7, 1e-7, 5e-8, 1e-8] (same as MeZO)
get_opt1_3b_lrs() {
    echo "1e-6 5e-7 1e-7 5e-8 1e-8"
}

# Mistral-7B: Grid search [5e-7, 1e-7, 5e-8, 1e-8, 5e-9] (same as MeZO)
get_mistral_lrs() {
    echo "5e-7 1e-7 5e-8 1e-8 5e-9"
}

# OPT-13B: Grid search [5e-6, 1e-6, 5e-7, 1e-7, 5e-8] (same as MeZO)
get_opt13b_lrs() {
    echo "5e-6 1e-6 5e-7 1e-7 5e-8"
}

# =============================================================================
# RoBERTa-Large Experiments (NO early stopping, grid search)
# =============================================================================
run_roberta() {
    local TASK=$1
    local LRS=$(get_roberta_lrs)
    local TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="lozo_roberta_${TASK_LOWER}"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[RoBERTa-Large] LOZO on ${TASK}"
        echo "  lr=${LR} (grid search)"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  NO early stopping"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_ROBERTA} \
            --task_name=${TASK} \
            --output_dir=result/lozo_roberta_${TASK}_lr${LR} \
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
            --trainer=lozo \
            --train_set_seed=0 \
            --lr_scheduler_type=constant \
            --eval_steps=${EVAL_STEPS} \
            --save_steps=${SAVE_STEPS} \
            --overwrite_output_dir \
            --metric_for_best_model=accuracy \
            --greater_is_better=true \
            --learning_rate=${LR} \
            --zo_eps=${ZO_EPS} \
            --lozo_rank=${LOZO_RANK} \
            --lozo_step_interval=${LOZO_STEP_INTERVAL} \
            --weight_decay=0 \
            --use_roberta_mlm

        echo "[RoBERTa-Large] ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# OPT-1.3B Experiments (with early stopping, grid search)
# =============================================================================
run_opt_1_3b() {
    local TASK=$1
    local LRS=$(get_opt1_3b_lrs)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="lozo_opt1.3b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[OPT-1.3B] LOZO on ${TASK}"
        echo "  lr=${LR} (grid search)"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  early_stopping=${EARLY_STOPPING}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_OPT_1_3B} \
            --task_name=${TASK} \
            --output_dir=result/lozo_opt1.3b_${TASK}_lr${LR} \
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
            --trainer=lozo \
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
            --lozo_rank=${LOZO_RANK} \
            --lozo_step_interval=${LOZO_STEP_INTERVAL} \
            --weight_decay=0

        echo "[OPT-1.3B] ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# Mistral-7B Experiments (fp16, with early stopping, grid search)
# =============================================================================
run_mistral_7b() {
    local TASK=$1
    local LRS=$(get_mistral_lrs)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="lozo_mistral7b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[Mistral-7B] LOZO on ${TASK} (fp16)"
        echo "  lr=${LR} (grid search)"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  early_stopping=${EARLY_STOPPING}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_MISTRAL_7B} \
            --task_name=${TASK} \
            --output_dir=result/lozo_mistral7b_${TASK}_lr${LR} \
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
            --trainer=lozo \
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
            --lozo_rank=${LOZO_RANK} \
            --lozo_step_interval=${LOZO_STEP_INTERVAL} \
            --weight_decay=0 \
            --load_float16

        echo "[Mistral-7B] ${TASK} lr=${LR} completed!"
        echo ""
    done
}

# =============================================================================
# OPT-13B Experiments (bf16, with early stopping, grid search)
# =============================================================================
run_opt_13b() {
    local TASK=$1
    local LRS=$(get_opt13b_lrs)
    read NUM_TRAIN NUM_DEV <<< $(get_data_params $TASK)

    export WANDB_PROJECT="lozo_opt13b_main"

    for LR in ${LRS}; do
        echo "======================================"
        echo "[OPT-13B] LOZO on ${TASK} (bf16)"
        echo "  lr=${LR} (grid search)"
        echo "  num_train=${NUM_TRAIN}, num_dev=${NUM_DEV}"
        echo "  early_stopping=${EARLY_STOPPING}"
        echo "======================================"

        python run.py \
            --model_name=${MODEL_OPT_13B} \
            --task_name=${TASK} \
            --output_dir=result/lozo_opt13b_${TASK}_lr${LR} \
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
            --trainer=lozo \
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
            --lozo_rank=${LOZO_RANK} \
            --lozo_step_interval=${LOZO_STEP_INTERVAL} \
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
echo "LOZO Main Experiments"
echo "============================================="
echo "Models: RoBERTa-Large, OPT-1.3B, Mistral-7B (fp16), OPT-13B (bf16)"
echo "LOZO params: rank=${LOZO_RANK}, step_interval=${LOZO_STEP_INTERVAL}"
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
    roberta)
        if [ "$TASK" == "all" ]; then
            for task in "${ROBERTA_DATASETS[@]}"; do
                run_roberta $task
            done
        else
            run_roberta $TASK
        fi
        ;;
    opt1.3b)
        if [ "$TASK" == "all" ]; then
            for task in "${LLM_DATASETS[@]}"; do
                run_opt_1_3b $task
            done
        else
            run_opt_1_3b $TASK
        fi
        ;;
    mistral7b)
        if [ "$TASK" == "all" ]; then
            for task in "${LLM_DATASETS[@]}"; do
                run_mistral_7b $task
            done
        else
            run_mistral_7b $TASK
        fi
        ;;
    opt13b)
        if [ "$TASK" == "all" ]; then
            for task in "${LLM_DATASETS[@]}"; do
                run_opt_13b $task
            done
        else
            run_opt_13b $TASK
        fi
        ;;
    all)
        echo "Running all models..."
        echo ""

        echo "=== RoBERTa-Large (grid search, no early stopping) ==="
        for task in "${ROBERTA_DATASETS[@]}"; do
            run_roberta $task
        done

        echo "=== OPT-1.3B ==="
        for task in "${LLM_DATASETS[@]}"; do
            run_opt_1_3b $task
        done

        echo "=== Mistral-7B (fp16) ==="
        for task in "${LLM_DATASETS[@]}"; do
            run_mistral_7b $task
        done

        echo "=== OPT-13B (bf16) ==="
        for task in "${LLM_DATASETS[@]}"; do
            run_opt_13b $task
        done
        ;;
    *)
        echo "============================================="
        echo "Usage: $0 [model] [task] [gpu_id]"
        echo ""
        echo "Models: roberta | opt1.3b | mistral7b | opt13b | all"
        echo ""
        echo "Tasks:"
        echo "  RoBERTa: SST2 | RTE | CB | WIC | TREC | all (5 tasks)"
        echo "  Others:  SST2 | RTE | WSC | Copa | TREC | WIC | all (6 tasks)"
        echo ""
        echo "Examples:"
        echo "  $0 roberta all 0        # RoBERTa on all 5 datasets, GPU 0"
        echo "  $0 opt13b SST2 1        # OPT-13B on SST2, GPU 1"
        echo "  $0 mistral7b all 2      # Mistral-7B on all 6 datasets, GPU 2"
        echo "  $0 all all 0            # All models on all datasets"
        echo "============================================="
        exit 1
        ;;
esac

echo "============================================="
echo "Experiments completed!"
echo "============================================="
echo ""
echo "Summary (learning rates same as MeZO):"
echo "  RoBERTa-Large: [1e-4, 1e-5, 1e-6], NO early stopping"
echo "  OPT-1.3B:      [1e-6, 5e-7, 1e-7, 5e-8, 1e-8], early stopping=${EARLY_STOPPING}"
echo "  Mistral-7B:    [5e-7, 1e-7, 5e-8, 1e-8, 5e-9] (fp16), early stopping=${EARLY_STOPPING}"
echo "  OPT-13B:       [5e-6, 1e-6, 5e-7, 1e-7, 5e-8] (bf16), early stopping=${EARLY_STOPPING}"
echo ""
echo "LOZO parameters: rank=${LOZO_RANK}, step_interval=${LOZO_STEP_INTERVAL}"
echo "============================================="

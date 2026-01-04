#!/bin/bash
# =============================================================================
# OPT-1.3B Ablation Study: Subspace Dimension k (k=m)
# Datasets: SST2, RTE
# k values: 1, 2, 4, 8
# =============================================================================

# Wandb settings
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
DATASETS=("SST2" "RTE")

# Learning rates
LRS=("1e-5" "5e-6" "1e-6" "5e-7" "1e-7")

# Subspace dimensions (k=m)
K_VALUES=(1 2 4 8)

# =============================================================================
# Run ablation for a specific k value
# =============================================================================
run_ablation() {
    local TASK=$1
    local K=$2
    local LR=$3

    echo "======================================"
    echo "Ablation: ${TASK}, k=${K}, lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="ablation_k${K}_${TASK,,}_opt1.3b"

    python run.py \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK} \
        --output_dir=result/ablation_${TASK}_k${K}_lr${LR} \
        --num_train_epochs=5 \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --load_best_model_at_end \
        --evaluation_strategy=steps \
        --save_strategy=steps \
        --save_total_limit=1 \
        --max_steps=${MAX_STEPS} \
        --logging_steps=10 \
        --num_eval=1000 \
        --num_train=1000 \
        --num_dev=500 \
        --train_as_classification \
        --trainer=bszo_v4 \
        --train_set_seed=0 \
        --lr_scheduler_type=constant \
        --eval_steps=${EVAL_STEPS} \
        --save_steps=${SAVE_STEPS} \
        --early_stopping_patience=${EARLY_STOPPING} \
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --bszo_fixed_subspace_dim=${K} \
        --bayesian_num_samples=${K} \
        --bayesian_adaptive_sampling=True \
        --bayesian_one_sided=True
}

# =============================================================================
# Main
# =============================================================================
echo "============================================="
echo "OPT-1.3B Ablation Study"
echo "Datasets: ${DATASETS[*]}"
echo "K values: ${K_VALUES[*]}"
echo "Learning rates: ${LRS[*]}"
echo "============================================="

# Parse arguments
TASK_ARG=${1:-"all"}
K_ARG=${2:-"all"}
LR_ARG=${3:-"all"}

if [ "$TASK_ARG" == "all" ]; then
    TASK_LIST=("${DATASETS[@]}")
else
    TASK_LIST=("$TASK_ARG")
fi

if [ "$K_ARG" == "all" ]; then
    K_LIST=("${K_VALUES[@]}")
else
    K_LIST=("$K_ARG")
fi

if [ "$LR_ARG" == "all" ]; then
    LR_LIST=("${LRS[@]}")
else
    LR_LIST=("$LR_ARG")
fi

# Run experiments
for task in "${TASK_LIST[@]}"; do
    for k in "${K_LIST[@]}"; do
        for lr in "${LR_LIST[@]}"; do
            run_ablation $task $k $lr
        done
    done
done

echo "============================================="
echo "Ablation study completed!"
echo "Run 'wandb sync' to upload results"
echo "============================================="

#!/bin/bash
# =============================================================================
# ZO-SGD experiments for OPT-13B
# Usage: ./run_zo_sgd_opt13b.sh [DATASET]
# Example: ./run_zo_sgd_opt13b.sh SST2
# =============================================================================

export WANDB_MODE=offline
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1

MODEL_NAME="/mnt/innovator/chl/models/opt-13b"
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
EARLY_STOPPING=8
ZO_EPS="1e-4"
LRS=("5e-6" "1e-6" "5e-7" "1e-7" "5e-8")

TASK=${1:-"SST2"}
DATASETS=("SST2" "RTE" "WSC" "Copa" "MultiRC")

run_task() {
    local TASK=$1
    export WANDB_PROJECT="zo_sgd_${TASK,,}_opt13b_ft"

    for lr in "${LRS[@]}"; do
        echo "======================================"
        echo "ZO-SGD | ${TASK} | lr=${lr}"
        echo "======================================"

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
            --num_eval=1000 \
            --num_train=1000 \
            --num_dev=500 \
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
            --weight_decay=0 \
            --load_bfloat16
    done
}

if [ "$TASK" == "all" ]; then
    for dataset in "${DATASETS[@]}"; do
        run_task $dataset
    done
else
    run_task $TASK
fi

echo "ZO-SGD experiments completed!"

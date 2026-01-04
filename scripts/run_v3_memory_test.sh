#!/bin/bash
# =============================================================================
# ZO Methods Memory Usage Test
# Models: OPT-1.3B, Mistral-7B, OPT-13B
# Methods: ZO-SGD, ZO-Adam, HiZOO, BSZO-V3
# Dataset: SST2
# Purpose: Record GPU memory usage for all ZO methods across different model sizes
# =============================================================================

# Wandb online mode to record memory
export WANDB_MODE=offline
export WANDB_DIR=/mnt/innovator/chl/wandb_cache
export WANDB_CACHE_DIR=/mnt/innovator/chl/wandb_cache

# Set cache directories
export HF_HOME=/mnt/innovator/chl/huggingface
export HF_DATASETS_CACHE=/mnt/innovator/chl/huggingface/datasets
export TRANSFORMERS_OFFLINE=1

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

# Common settings
TASK="SST2"
BATCH_SIZE=16
MAX_STEPS=600
EVAL_STEPS=200
SAVE_STEPS=200
ZO_EPS="1e-4"
NUM_TRAIN=1000
NUM_DEV=500

# Model paths
MODEL_OPT_1_3B="/mnt/innovator/chl/models/opt-1.3b"
MODEL_MISTRAL_7B="/mnt/innovator/chl/models/mistral-7b"
MODEL_OPT_13B="/mnt/innovator/chl/models/opt-13b"

echo "============================================="
echo "ZO Methods Memory Usage Test"
echo "Models: OPT-1.3B, Mistral-7B, OPT-13B"
echo "Methods: ZO-SGD, ZO-Adam, HiZOO, BSZO-V3"
echo "Dataset: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Steps: ${MAX_STEPS}"
echo "Single GPU mode (CUDA_VISIBLE_DEVICES=0)"
echo "============================================="

# =============================================================================
# OPT-1.3B Methods
# =============================================================================

run_opt_1_3b_zo_sgd() {
    local LR="1e-6"
    echo "======================================"
    echo "[OPT-1.3B] Running ZO-SGD with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_1_3B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt1.3b_zo_sgd_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0
}

run_opt_1_3b_zo_adam() {
    local LR="1e-5"
    echo "======================================"
    echo "[OPT-1.3B] Running ZO-Adam with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_1_3B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt1.3b_zo_adam_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0
}

run_opt_1_3b_hizoo() {
    local LR="1e-6"
    echo "======================================"
    echo "[OPT-1.3B] Running HiZOO with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_1_3B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt1.3b_hizoo_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0
}

run_opt_1_3b_bszo_v3() {
    local LR="1e-6"
    echo "======================================"
    echo "[OPT-1.3B] Running BSZO-V3 with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_1_3B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt1.3b_bszo_v3_${TASK} \
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
        --weight_decay=0
}

# =============================================================================
# Mistral-7B Methods
# =============================================================================

run_mistral_7b_zo_sgd() {
    local LR="1e-6"
    echo "======================================"
    echo "[Mistral-7B] Running ZO-SGD with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_MISTRAL_7B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/mistral7b_zo_sgd_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_float16
}

run_mistral_7b_zo_adam() {
    local LR="1e-5"
    echo "======================================"
    echo "[Mistral-7B] Running ZO-Adam with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_MISTRAL_7B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/mistral7b_zo_adam_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_float16
}

run_mistral_7b_hizoo() {
    local LR="1e-6"
    echo "======================================"
    echo "[Mistral-7B] Running HiZOO with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_MISTRAL_7B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/mistral7b_hizoo_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_float16
}

run_mistral_7b_bszo_v3() {
    local LR="1e-6"
    echo "======================================"
    echo "[Mistral-7B] Running BSZO-V3 with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_MISTRAL_7B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/mistral7b_bszo_v3_${TASK} \
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
        --load_float16
}

# =============================================================================
# OPT-13B Methods
# =============================================================================

run_opt_13b_zo_sgd() {
    local LR="1e-6"
    echo "======================================"
    echo "[OPT-13B] Running ZO-SGD with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_13B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt13b_zo_sgd_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_bfloat16
}

run_opt_13b_zo_adam() {
    local LR="1e-5"
    echo "======================================"
    echo "[OPT-13B] Running ZO-Adam with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_13B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt13b_zo_adam_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_bfloat16
}

run_opt_13b_hizoo() {
    local LR="1e-6"
    echo "======================================"
    echo "[OPT-13B] Running HiZOO with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_13B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt13b_hizoo_${TASK} \
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
        --metric_for_best_model=accuracy \
        --greater_is_better=true \
        --learning_rate=${LR} \
        --zo_eps=${ZO_EPS} \
        --weight_decay=0 \
        --load_bfloat16
}

run_opt_13b_bszo_v3() {
    local LR="1e-6"
    echo "======================================"
    echo "[OPT-13B] Running BSZO-V3 with lr=${LR}"
    echo "======================================"

    export WANDB_PROJECT="memory_test_all_methods"

    python run.py \
        --model_name=${MODEL_OPT_13B} \
        --task_name=${TASK} \
        --output_dir=result/memory_test/opt13b_bszo_v3_${TASK} \
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
        --load_bfloat16
}

# =============================================================================
# Helper functions to run all methods for a model
# =============================================================================

run_opt_1_3b_all() {
    run_opt_1_3b_zo_sgd
    run_opt_1_3b_zo_adam
    run_opt_1_3b_hizoo
    run_opt_1_3b_bszo_v3
}

run_mistral_7b_all() {
    run_mistral_7b_zo_sgd
    run_mistral_7b_zo_adam
    run_mistral_7b_hizoo
    run_mistral_7b_bszo_v3
}

run_opt_13b_all() {
    run_opt_13b_zo_sgd
    run_opt_13b_zo_adam
    run_opt_13b_hizoo
    run_opt_13b_bszo_v3
}

# =============================================================================
# Main: Parse arguments and run
# =============================================================================

# Usage: ./run_v3_memory_test.sh [model] [method]
# model: opt1.3b | mistral7b | opt13b | all
# method: zo_sgd | zo_adam | hizoo | bszo_v3 | all

MODEL=${1:-"all"}
METHOD=${2:-"all"}

run_single() {
    local model=$1
    local method=$2

    case "${model}_${method}" in
        opt1.3b_zo_sgd)     run_opt_1_3b_zo_sgd ;;
        opt1.3b_zo_adam)    run_opt_1_3b_zo_adam ;;
        opt1.3b_hizoo)      run_opt_1_3b_hizoo ;;
        opt1.3b_bszo_v3) run_opt_1_3b_bszo_v3 ;;
        opt1.3b_all)        run_opt_1_3b_all ;;

        mistral7b_zo_sgd)     run_mistral_7b_zo_sgd ;;
        mistral7b_zo_adam)    run_mistral_7b_zo_adam ;;
        mistral7b_hizoo)      run_mistral_7b_hizoo ;;
        mistral7b_bszo_v3) run_mistral_7b_bszo_v3 ;;
        mistral7b_all)        run_mistral_7b_all ;;

        opt13b_zo_sgd)     run_opt_13b_zo_sgd ;;
        opt13b_zo_adam)    run_opt_13b_zo_adam ;;
        opt13b_hizoo)      run_opt_13b_hizoo ;;
        opt13b_bszo_v3) run_opt_13b_bszo_v3 ;;
        opt13b_all)        run_opt_13b_all ;;

        *)
            echo "Invalid combination: model=${model}, method=${method}"
            return 1
            ;;
    esac
}

case $MODEL in
    opt1.3b|mistral7b|opt13b)
        run_single $MODEL $METHOD
        ;;
    all)
        if [ "$METHOD" == "all" ]; then
            run_opt_1_3b_all
            run_mistral_7b_all
            run_opt_13b_all
        else
            run_single opt1.3b $METHOD
            run_single mistral7b $METHOD
            run_single opt13b $METHOD
        fi
        ;;
    *)
        echo "============================================="
        echo "Usage: $0 [model] [method]"
        echo ""
        echo "Models:  opt1.3b | mistral7b | opt13b | all"
        echo "Methods: zo_sgd | zo_adam | hizoo | bszo_v3 | all"
        echo ""
        echo "Examples:"
        echo "  $0 all all           # Run all models, all methods"
        echo "  $0 opt1.3b all       # Run all methods on OPT-1.3B"
        echo "  $0 all zo_sgd        # Run ZO-SGD on all models"
        echo "  $0 mistral7b hizoo   # Run HiZOO on Mistral-7B"
        echo "============================================="
        exit 1
        ;;
esac

echo "============================================="
echo "Memory test completed!"
echo "Check wandb for GPU memory usage"
echo "============================================="

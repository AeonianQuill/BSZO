#!/bin/bash
# =============================================================================
# V3 vs V4 Equivalence Test
# 目的：验证 v3 的缓存优化和 v4 的重新计算是否数学等价
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
TASK="WIC"
BATCH_SIZE=16
MAX_STEPS=20000
EVAL_STEPS=500
SAVE_STEPS=500
EARLY_STOPPING=8
ZO_EPS="1e-4"
LR="1e-6"
SEED=0

# 数据集大小
NUM_TRAIN=450
NUM_DEV=100
NUM_EVAL=1000

# 核心参数：v3 和 v4 使用完全相同的配置
SUBSPACE_DIM=2
NUM_SAMPLES=3
ADAPTIVE_SAMPLING=True
ONE_SIDED=True

# =============================================================================
# Run V3
# =============================================================================
echo "======================================"
echo "Running BSZO V3"
echo "  subspace_dim=${SUBSPACE_DIM}"
echo "  num_samples=${NUM_SAMPLES}"
echo "  adaptive_sampling=${ADAPTIVE_SAMPLING}"
echo "  one_sided=${ONE_SIDED}"
echo "======================================"

export WANDB_PROJECT="v3_v4_equivalence_test"

python run.py \
    --model_name=${MODEL_NAME} \
    --task_name=${TASK} \
    --output_dir=result/equiv_v3_opt1.3b_${TASK}_lr${LR} \
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
    --train_set_seed=${SEED} \
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
    --load_float16 \
    --bszo_fixed_subspace_dim=${SUBSPACE_DIM} \
    --bayesian_num_samples=${NUM_SAMPLES} \
    --bayesian_adaptive_sampling=${ADAPTIVE_SAMPLING} \
    --bayesian_one_sided=${ONE_SIDED}

# =============================================================================
# Run V4
# =============================================================================
echo "======================================"
echo "Running BSZO V4"
echo "  subspace_dim=${SUBSPACE_DIM}"
echo "  num_samples=${NUM_SAMPLES}"
echo "  adaptive_sampling=${ADAPTIVE_SAMPLING}"
echo "  one_sided=${ONE_SIDED}"
echo "======================================"

python run.py \
    --model_name=${MODEL_NAME} \
    --task_name=${TASK} \
    --output_dir=result/equiv_v4_opt1.3b_${TASK}_lr${LR} \
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
    --trainer=bszo_v4 \
    --train_set_seed=${SEED} \
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
    --load_float16 \
    --bszo_fixed_subspace_dim=${SUBSPACE_DIM} \
    --bayesian_num_samples=${NUM_SAMPLES} \
    --bayesian_adaptive_sampling=${ADAPTIVE_SAMPLING} \
    --bayesian_one_sided=${ONE_SIDED}

echo "======================================"
echo "V3 vs V4 equivalence test completed!"
echo ""
echo "Compare results:"
echo "  V3: result/equiv_v3_opt1.3b_${TASK}_lr${LR}"
echo "  V4: result/equiv_v4_opt1.3b_${TASK}_lr${LR}"
echo ""
echo "If equivalent, curves should be within random noise."
echo "======================================"

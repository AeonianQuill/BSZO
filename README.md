# Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer

A derivative-free optimization method for LLMs based on Bayesian inference in subspace, achieving strong performance and great robust with low memory cost.

## Features

- Bayesian gradient estimation in random k-dim subspace
- Memory efficient: O(n + k^2)
- Supports OPT, RoBERTa-large, Mistral models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py \
    --model_name=facebook/opt-1.3b \
    --task_name=SST2 \
    --output_dir=result/bszo_v3_opt1.3b_SST2 \
    --num_train_epochs=5 \
    --per_device_train_batch_size=16 \
    --load_best_model_at_end \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --save_total_limit=1 \
    --max_steps=20000 \
    --logging_steps=10 \
    --eval_steps=500 \
    --save_steps=500 \
    --num_train=1000 \
    --num_dev=500 \
    --num_eval=1000 \
    --train_as_classification \
    --trainer=bszo_v3 \
    --train_set_seed=0 \
    --lr_scheduler_type=constant \
    --overwrite_output_dir \
    --early_stopping_patience=8 \
    --metric_for_best_model=accuracy \
    --greater_is_better=true \
    --learning_rate=1e-7 \
    --zo_eps=1e-4 \
    --bszo_fixed_subspace_dim=2 \
    --bayesian_num_samples=3 \
    --bayesian_adaptive_sampling=True \
    --bayesian_one_sided=True
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--trainer` | `bszo_v3` or `bszo_v4` | - |
| `--bszo_fixed_subspace_dim` | Subspace dimension | 2 |
| `--bayesian_num_samples` | Samples per step | 3 |
| `--bayesian_one_sided` | One-sided difference | True |

## Others
See for more details. More will be added soon.

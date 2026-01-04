# BSZO: Bayesian Subspace Zeroth-Order Optimizer

A memory-efficient zeroth-order optimizer for LLM fine-tuning.

## Features

- Bayesian gradient estimation in random k-dim subspace
- Memory efficient: O(k) instead of O(n)
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
    --trainer=bszo_v3 \
    --learning_rate=1e-7 \
    --bszo_fixed_subspace_dim=2 \
    --bayesian_num_samples=3
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--trainer` | `bszo_v3` or `bszo_v4` | - |
| `--bszo_fixed_subspace_dim` | Subspace dimension | 2 |
| `--bayesian_num_samples` | Samples per step | 3 |
| `--bayesian_one_sided` | One-sided difference | True |

## Scripts

See `scripts/` for experiment configurations.


"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import inspect
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Union, Any
from typing import TYPE_CHECKING, Optional, Iterable

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
)
# is_fairscale_available was removed in newer transformers versions
try:
    from transformers.integrations import is_fairscale_available
except ImportError:
    def is_fairscale_available():
        return False
# Handle different transformers versions
try:
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
except ImportError:
    is_torch_greater_or_equal_than_1_10 = version.parse(torch.__version__) >= version.parse("1.10")
    is_torch_less_than_1_11 = version.parse(torch.__version__) < version.parse("1.11")
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
# ShardedDDPOption was removed in newer transformers versions
try:
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    from enum import Enum
    class ShardedDDPOption(str, Enum):
        SIMPLE = "simple"
        ZERO_DP_2 = "zero_dp_2"
        ZERO_DP_3 = "zero_dp_3"
        OFFLOAD = "offload"
from transformers.utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

# from torch.optim.optimizer import StateDict, params_t
import wandb
# from gradient_pruning.pruning_utils import (
#     fast_random_mask_like,
#     estimate_pretrained_model_magnitude_pruning_threshold,
#     compute_named_parameters_to_sparsity,
# )
from metrics import f1
from utils import OPT_PERTURBATION_LEVEL_TO_REGEX

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    # ZO-Bench added: new parameters to our traininer
    def __init__(self, evaluate_func, dev_samples, eval_samples, perturb_module_regex, framework=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class

        # Compatibility fix for newer transformers versions (4.41+)
        # do_grad_scaling was removed/changed in newer versions
        if not hasattr(self, 'do_grad_scaling'):
            self.do_grad_scaling = False

        self.evaluate_func = evaluate_func
        self.dev_samples = dev_samples
        self.eval_samples = eval_samples
        self.perturb_module_regex = perturb_module_regex
        self.framework = framework  # Store reference to Framework instance

        # Track best test accuracy corresponding to best validation accuracy
        # This avoids relying on load_best_model_at_end which has issues with sharded safetensors
        self.best_test_acc = None

        # Compatibility: detect if _maybe_log_save_evaluate has grad_norm parameter
        sig = inspect.signature(super()._maybe_log_save_evaluate)
        self._has_grad_norm_param = 'grad_norm' in sig.parameters

    def _compat_maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Compatibility wrapper for _maybe_log_save_evaluate across transformers versions."""
        # Manual logging check for compatibility with newer transformers
        logging_steps = getattr(self.args, 'logging_steps', None)
        if logging_steps and self.state.global_step % logging_steps == 0 and self.state.global_step > 0:
            # Calculate average loss since last log
            logs = {}
            tr_loss_scalar = tr_loss.item() if hasattr(tr_loss, 'item') else tr_loss

            # Compute average loss since last log (not cumulative)
            steps_since_last_log = self.state.global_step - self._globalstep_last_logged
            if steps_since_last_log > 0:
                # Get loss since last log
                loss_since_last_log = tr_loss_scalar - getattr(self, '_last_logged_loss', 0.0)
                logs["loss"] = round(loss_since_last_log / steps_since_last_log, 4)
            else:
                logs["loss"] = round(tr_loss_scalar, 4)

            # Update tracking variables
            self._last_logged_loss = tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            logs["learning_rate"] = self._get_learning_rate() if hasattr(self, '_get_learning_rate') else self.args.learning_rate
            logs["epoch"] = round(epoch, 2)
            logs["global_step"] = self.state.global_step
            # Add train/ prefix for wandb dashboard
            logs["train/loss"] = logs["loss"]
            logs["train/learning_rate"] = logs["learning_rate"]
            print(logs)
            # Log via self.log (WandbCallback will handle wandb logging)
            self.log(logs)

        # Manual evaluation check for compatibility with newer transformers
        eval_steps = getattr(self.args, 'eval_steps', None)
        eval_strategy = getattr(self.args, 'eval_strategy', None) or getattr(self.args, 'evaluation_strategy', 'no')

        if str(eval_strategy).lower() in ['steps', 'intervalstrategy.steps'] and eval_steps and self.state.global_step % eval_steps == 0 and self.state.global_step > 0:
            # Force evaluation at eval_steps intervals
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Call on_evaluate FIRST so EarlyStoppingCallback can compare current metric with best_metric
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            # Log metrics
            print(f"[Eval at step {self.state.global_step}] {metrics}")
            # Add eval/ prefix keys for wandb dashboard
            metrics["eval/val_acc"] = metrics.get("val_acc", metrics.get("eval_accuracy"))
            metrics["eval/test_acc"] = metrics.get("test_acc", metrics.get("eval_test_accuracy"))
            # Log via self.log only (WandbCallback will handle wandb logging)
            self.log(metrics)

            # Update state.best_metric and state.best_model_checkpoint AFTER on_evaluate
            # (this is normally done in _save_checkpoint)
            if self.args.load_best_model_at_end:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics.get(metric_to_check)

                if metric_value is not None:
                    is_new_best = (
                        self.state.best_metric is None
                        or (self.args.greater_is_better and metric_value > self.state.best_metric)
                        or (not self.args.greater_is_better and metric_value < self.state.best_metric)
                    )
                    if is_new_best:
                        self.state.best_metric = metric_value
                        # Record the test accuracy corresponding to best validation accuracy
                        # This avoids relying on load_best_model_at_end which has issues with sharded safetensors
                        self.best_test_acc = metrics.get("test_acc") or metrics.get("eval_test_accuracy")
                        # Save checkpoint when we find a new best model
                        # This ensures the checkpoint actually exists when we try to load it later
                        self._save_checkpoint(model, trial, metrics=metrics)
                        print(f"[Best metric updated] {metric_to_check}={metric_value}, best_test_acc={self.best_test_acc}")

        # Create a zero tensor to prevent parent class from logging duplicate loss
        # (we already logged the correct averaged loss above)
        zero_loss = tr_loss * 0.0

        if self._has_grad_norm_param:
            # Newer transformers (4.41+) has grad_norm parameter
            self._maybe_log_save_evaluate(zero_loss, None, model, trial, epoch, ignore_keys_for_eval)
        else:
            # Older transformers
            self._maybe_log_save_evaluate(zero_loss, model, trial, epoch, ignore_keys_for_eval)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate method to use our custom evaluation function and return metrics
        with the correct keys for HuggingFace's load_best_model_at_end functionality.
        """
        # Call the custom evaluation function on dev samples (for validation)
        val_metrics = self.evaluate_func([], self.dev_samples)
        # Call the custom evaluation function on eval samples (for testing)
        test_metrics = self.evaluate_func([], self.eval_samples)

        # Prepare the metrics dictionary with appropriate prefixes
        metrics = {}

        if "accuracy" in test_metrics:
            # Add metrics with eval_ prefix for HuggingFace compatibility
            metrics[f"{metric_key_prefix}_accuracy"] = val_metrics["accuracy"]
            metrics[f"{metric_key_prefix}_test_accuracy"] = test_metrics["accuracy"]
            # Also add without prefix for backwards compatibility
            metrics["test_acc"] = test_metrics["accuracy"]
            metrics["val_acc"] = val_metrics["accuracy"]
            # IMPORTANT: Add eval_loss for HuggingFace checkpoint saving
            # Convert accuracy to loss (higher accuracy = lower loss)
            metrics[f"{metric_key_prefix}_loss"] = 1.0 - val_metrics["accuracy"]
        else:
            # Handle other metric types
            keys = list(test_metrics.keys())
            for k in keys:
                metrics[f"{metric_key_prefix}_{k}"] = val_metrics[k]
                metrics[f"{metric_key_prefix}_test_{k}"] = test_metrics[k]
                metrics[f"test_{k}"] = test_metrics[k]
                metrics[f"val_{k}"] = val_metrics[k]
            # Add eval_loss for non-accuracy metrics (use first metric as proxy)
            first_metric_value = val_metrics[keys[0]]
            # Assume lower is better for most metrics, if not adjust here
            metrics[f"{metric_key_prefix}_loss"] = first_metric_value

        # Call the parent's evaluate to get standard metrics (loss, etc.)
        if eval_dataset is not None:
            parent_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            # Merge with our custom metrics (our metrics take precedence)
            parent_metrics.update(metrics)
            return parent_metrics

        return metrics

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()  # ----newly-added

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type in ["opt", "llama", "mistral"]:
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(train_dataloader):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2", "llama", "mistral"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # Compatibility: sharded_ddp and fsdp were removed in newer transformers versions
        sharded_ddp = getattr(self, 'sharded_ddp', None)
        fsdp = getattr(self, 'fsdp', None)
        delay_optimizer_creation = (
                sharded_ddp is not None
                and sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # overload the optimizer here
        if args.trainer == "zo_adam":
            self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "zo_sgd":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            # self.optimizer = {name: SGD([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
            # print(f"### args.lr_scheduler: {args.lr_scheduler_type}")
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "hizoo":
            # HiZOO: Hessian-Informed Zeroth-Order Optimization
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            # HiZOO specific parameters
            # Note: official uses smooth=1e-8 with eps=1e-3
            # For eps=1e-4, need smooth=1e-10 to match (smooth/eps² ratio)
            self.hizoo_smooth = getattr(args, 'hizoo_smooth', 1e-10)  # Hessian smooth coefficient (alpha)
            self.hizoo_eps = getattr(args, 'hizoo_eps', 1e-8)  # Epsilon for numerical stability
            self.hizoo_v = {}  # Hessian diagonal (per parameter)
            self.hizoo_step_count = 0
            logger.info(f"HiZOO initialized with lr={args.learning_rate}, hessian_smooth={self.hizoo_smooth}, eps={self.hizoo_eps}")
        elif args.trainer == "lozo":
            # LOZO: Low-rank Zeroth-Order Optimization
            # Reference: https://github.com/optsuite/LOZO
            # Note: step counter and v matrices are initialized in lozo_step (matches official)
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            # LOZO specific parameters (used in lozo_step/lozo_perturb_parameters/lozo_update)
            self.lozo_rank = getattr(args, 'lozo_rank', 2)  # rank r for low-rank perturbation
            self.lozo_step_interval = getattr(args, 'lozo_step_interval', 50)  # ν: how often to resample v
            logger.info(f"LOZO initialized with lr={args.learning_rate}, rank={self.lozo_rank}, step_interval={self.lozo_step_interval}")
        elif args.trainer in ["bszo_v3", "bszo_v4"]:
            # BSZO (Bayesian Subspace Zeroth-Order) optimizer initialization
            if args.trainer == "bszo_v3":
                from bszo_optimizer_v3 import BSZOOptimizer
                logger.info("Using BSZO V3 (Bayesian Subspace Zeroth-Order)")
            else:  # bszo_v4
                from bszo_optimizer_v4 import BSZOOptimizer
                logger.info("Using BSZO V4 (Bayesian Subspace Zeroth-Order)")
            logger.info(f"  - Subspace dim: {getattr(args, 'bszo_fixed_subspace_dim', 2)}")
            logger.info(f"  - Num samples: {getattr(args, 'bayesian_num_samples', 4)}")
            logger.info(f"  - Sigma prior: {getattr(args, 'bayesian_sigma_prior', 'auto')}")
            logger.info(f"  - Sigma noise: {getattr(args, 'bayesian_sigma_noise', 'auto')}")
            logger.info(f"  - Adaptive sampling: {getattr(args, 'bayesian_adaptive_sampling', True)}")

            self.named_parameters_to_optim = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((name, param))

            # Create validation function for optimizer
            def validation_loss_fn(model, _):
                """Evaluate on validation set and return average loss"""
                val_metrics = self.evaluate_func([], self.dev_samples)
                # Return loss (lower is better)
                if "accuracy" in val_metrics:
                    # Convert accuracy to loss: loss = 1 - accuracy
                    return 1.0 - val_metrics["accuracy"]
                else:
                    # Assume the first metric is the loss
                    return list(val_metrics.values())[0]

            # BSZO doesn't use validation rollback
            self.bszo_optimizer = BSZOOptimizer(
                self.named_parameters_to_optim,
                self.zo_forward,
                args,
                eval_fn=None
            )
            logger.info(f"BSZO initialized with {len(self.named_parameters_to_optim)} parameter groups")
        else:
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.optimizer == "adam":
                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # Main training loop
        total_steps = 0
        total_batches = 0  # ZO-Bench added: batch-level counter for eval_batch
        self.sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_sparsity = None  # None, float, or dict

        if self.args.sparse_gradient_group == "layer" or self.args.gradient_sparsity is None:
            self.gradient_sparsity = self.args.gradient_sparsity
            print(f"### layer-wise gradient sparsity = {self.gradient_sparsity}")
        elif self.args.sparse_gradient_group == "global":
            threshold = estimate_pretrained_model_magnitude_pruning_threshold(model, self.args.gradient_sparsity)
            self.gradient_sparsity = compute_named_parameters_to_sparsity(model, threshold)
            print(f"### global gradient sparsity, weight magnitude threshold = {threshold}")

        for epoch in range(epochs_trained, num_train_epochs):
            print(f"-------------------------- Training Epoch {epoch} --------------------------")
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # Start one epoch training
            for step, inputs in enumerate(epoch_iterator):

                if total_steps % self.args.sparse_gradient_resample_steps == 0:
                    self.sparse_grad_random_seed = np.random.randint(1000000000)

                total_steps += 1

                # torch.cuda.synchronize()
                step_start_time = time.time()

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                if args.trainer in ["zo_sgd", "zo_adam"]:
                    if args.module_wise_perturbation:
                        assert args.q == 1, "module-wise perturbation only supports q=1"
                        if args.coordinate_perturbation:
                            tr_loss_step = self.zo_step_with_module_wise_perturbation_coordinate(model, inputs)
                        else:
                            tr_loss_step = self.zo_step_with_module_wise_perturbation(model, inputs)
                    elif args.q == 1:
                        tr_loss_step = self.zo_step(model, inputs)
                    elif args.q > 1:
                        tr_loss_step = self.zo_step_v1(model, inputs)
                    else:
                        raise ValueError(f"q={args.q} is not supported.")
                elif args.trainer == "hizoo":
                    tr_loss_step = self.hizoo_step(model, inputs)
                elif args.trainer == "lozo":
                    tr_loss_step = self.lozo_step(model, inputs)
                elif args.trainer in ["bszo_v3", "bszo_v4"]:
                    tr_loss_step = self.bszo_optimizer.step(model, inputs)
                    total_batches += 1  # increment batch counter after BSZO step
                else:
                    if (
                            ((step + 1) % args.gradient_accumulation_steps != 0)
                            and args.local_rank != -1
                            and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (math.isnan(tr_loss_step) or math.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer in ["zo_sgd", "zo_adam"]:
                        self.zo_update(model)
                    elif args.trainer == "hizoo":
                        pass  # HiZOO updates parameters in hizoo_step directly
                    elif args.trainer == "lozo":
                        pass  # LOZO updates parameters in lozo_step directly
                    else:
                        self.gradient_update(model)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._compat_maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # torch.cuda.synchronize()
                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                max_memory_allocated = 0
                for device_id in range(torch.cuda.device_count()):
                    # this is not accurate since max memory does not happen simultaneously across all devices
                    max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                # Log via self.log only (WandbCallback will handle wandb logging)
                self.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                          "step_consumption": train_step_duration * 1000})

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._compat_maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1 and dist.is_initialized():
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        # Log via self.log only (WandbCallback will handle wandb logging)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def gradient_update(self, model):
        args = self.args

        # Gradient clipping
        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
            # deepspeed does its own clipping

            if self.do_grad_scaling:
                # Reduce gradients first for XLA
                if is_torch_tpu_available():
                    gradients = xm._fetch_gradients(self.optimizer)
                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                # AMP: gradients need unscaling
                self.scaler.unscale_(self.optimizer)

            if is_sagemaker_mp_enabled() and args.fp16:
                self.optimizer.clip_master_grads(args.max_grad_norm)
            elif hasattr(self.optimizer, "clip_grad_norm"):
                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                self.optimizer.clip_grad_norm(args.max_grad_norm)
            elif hasattr(model, "clip_grad_norm_"):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                model.clip_grad_norm_(args.max_grad_norm)
            else:
                # Revert to normal clipping otherwise, handling Apex or full precision
                nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                    args.max_grad_norm,
                )

        # Optimizer step
        optimizer_was_run = True
        if self.deepspeed:
            pass  # called outside the loop
        elif is_torch_tpu_available():
            if self.do_grad_scaling:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                xm.optimizer_step(self.optimizer)
        elif self.do_grad_scaling:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()

        if optimizer_was_run and not self.deepspeed:
            self.lr_scheduler.step()
        model.zero_grad()

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def get_grad_sparsity_by_name(self, name):
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # Sparse gradient
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            if grad_sparsity is not None:
                param.grad[fast_random_mask_like(param.grad, grad_sparsity, generator=self.sparse_grad_rng)] = 0

        return loss.detach()

    @staticmethod
    def grouped_module_iter(model: nn.Module, group_level: str) -> Iterable[nn.Module]:
        """
        Iterate over the top-level modules of a model and yield groups of modules that should be trained together.

        Parameters
        ----------
        model: nn.Module
            The model to iterate over.
        group_level: str
            The level at which to iterate over the model. One of "transformer", "mlp-attn", "linear"
        """
        reg_pattern = OPT_PERTURBATION_LEVEL_TO_REGEX[group_level]
        for name, module in model.named_modules():
            if re.match(reg_pattern, name):
                yield name, module

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation_coordinate(self, model, inputs):
        """Update the parameters right after perturbing the parameters."""
        args = self.args
        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            all_losses.append(loss1)

            for _ in range(args.q):
                if self.args.perturbation_mode == "one_side":
                    self.zo_perturb_parameters(scaling_factor=-1)
                    loss2 = self.zo_forward(model, inputs)
                    self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
                else:  # two side perturbation
                    self.zo_perturb_parameters(scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs)
                    self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                    # Reset model back to its parameters at start of step
                    self.zo_perturb_parameters(scaling_factor=1)

                # Set the random seed to ensure that we sample the same z for perturbation/update
                torch.manual_seed(self.zo_random_seed)
                self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
                for name, param in self.named_parameters_to_optim:
                    # Resample z
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                     dtype=param.data.dtype)
                    grad_sparsity = self.get_grad_sparsity_by_name(name)
                    if grad_sparsity is not None:
                        z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                    graddiff_times_z = self.projected_grad * z

                    param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                    self.optimizer.step()  # will only update grad that is not None.
                    param.grad = None  # avoid further update.

        assert self.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation(self, model, inputs):
        """Update all parameters once after perturbing all the parameters."""
        args = self.args
        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []
        module_name_to_projected_grads = {}

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            all_losses.append(loss1)

            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            module_name_to_projected_grads[module_name] = self.projected_grad

        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.projected_grad = module_name_to_projected_grads[module_name]

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                self.optimizer.step()  # will only update grad that is not None.
                param.grad = None  # avoid further update.

        assert self.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                # Fix: convert to Python float FIRST to avoid FP16 overflow in division
                # (same fix as HiZOO: loss1.item() - loss2.item() instead of (loss1 - loss2).item())
                self.projected_grad = (loss1.item() - loss2.item()) / self.args.zo_eps
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                # Fix: convert to Python float FIRST to avoid FP16 overflow in division
                self.projected_grad = (loss1.item() - loss2.item()) / (2 * self.args.zo_eps)

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q
                self.optimizer.step()
                param.grad = None  # avoid further update

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def hizoo_step(self, model, inputs):
        """
        HiZOO: Hessian-Informed Zeroth-Order Optimization

        Follows official implementation: https://github.com/Yanjun-Zhao/HiZOO

        Key changes from previous version:
        - Use tensor operations instead of .item() for numerical consistency
        - Match official computation order exactly
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        random_seed = np.random.randint(1000000000)

        # Initialize Hessian matrix if needed (official: init to 1)
        if not self.hizoo_v:
            for name, param in self.named_parameters_to_optim:
                self.hizoo_v[name] = torch.ones(size=param.data.size(),
                                                 device=param.data.device,
                                                 dtype=param.data.dtype)

        mu = args.zo_eps  # perturbation scale
        alpha = self.hizoo_smooth  # Hessian smooth coefficient

        with torch.no_grad():
            # 1. Baseline loss at original parameters
            loss_original = self.zo_forward(model, inputs)

            # 2. First function evaluation: θ + ε/√H * z
            self._hizoo_perturb_parameters_official(random_seed, scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # 3. Second function evaluation: θ - ε/√H * z
            self._hizoo_perturb_parameters_official(random_seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            # 4. Reset to original parameters
            self._hizoo_perturb_parameters_official(random_seed, scaling_factor=1)

            # 5. Compute loss differences (handle multi-GPU: losses may be on different devices)
            # Convert to Python scalars to avoid device mismatch, then back to tensor for computation
            loss_diff_hess = abs(loss1.item() + loss2.item() - 2 * loss_original.item())
            loss_diff_grad = loss1.item() - loss2.item()

            # 6. Update Hessian and parameters (official computation order)
            torch.manual_seed(random_seed)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                               device=param.data.device, dtype=param.data.dtype)

                # Official Hessian update:
                # H_temp = H * z²
                # H_estimator = |loss1 + loss2 - 2*loss0| * H_temp * alpha / (2*eps²)
                # H_new = (1-alpha) * H + H_estimator
                Hessian_temp = self.hizoo_v[name] * z * z
                Hessian_estimator = loss_diff_hess * Hessian_temp * alpha / (2 * mu * mu)

                self.hizoo_v[name] = (1 - alpha) * self.hizoo_v[name] + Hessian_estimator

                # Official gradient update:
                # grad = (loss1 - loss2) / (2*eps) * z / √H
                # param = param - lr * (grad + wd * param)
                grad = loss_diff_grad / (2 * mu) * z / torch.sqrt(self.hizoo_v[name])
                param.data = param.data - args.learning_rate * (grad + args.weight_decay * param.data)

        return loss_original

    def _hizoo_perturb_parameters_official(self, random_seed, scaling_factor=1):
        """
        Official HiZOO perturbation: θ = θ + scaling_factor / √H * z * ε

        Note: Official version does NOT add eps to denominator
        """
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(),
                           device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor / torch.sqrt(self.hizoo_v[name]) * z * self.args.zo_eps

    ############## LOZO (Low-rank Zeroth-Order Optimization) ##############
    # Reference: https://github.com/optsuite/LOZO
    # This implementation follows the official LOZO code exactly

    def lozo_perturb_parameters(self, scaling_factor=1):
        """
        Perturb the parameters with random vector uv^t.
        Exactly matches official LOZO implementation.
        """
        args = self.args
        step = self.lozo_step_count
        rank_r = self.lozo_rank

        torch.manual_seed(self.lozo_random_seed)

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                # v is resampled every step_interval steps
                if step % self.lozo_step_interval == 0:
                    v = torch.randn(param.data.size(1), rank_r,
                                   device=param.data.device, dtype=param.data.dtype)
                    self.lozo_v[name] = v
                else:
                    v = self.lozo_v[name]

                # u is resampled every step (using random_gaussian_matrix style)
                u = torch.randn(param.data.size(0), rank_r,
                               device=param.data.device, dtype=param.data.dtype)

                # Perturbation: θ + scaling_factor * (u @ v^T) * ε
                param.data = param.data + scaling_factor * (u @ v.t()) * args.zo_eps
            else:
                # 1D parameters (bias): use standard Gaussian noise
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * args.zo_eps

    def lozo_update(self):
        """
        Update parameters with the estimated gradient.
        Exactly matches official LOZO implementation.
        """
        args = self.args
        rank_r = self.lozo_rank

        # Reset the random seed for sampling (same as in perturbation)
        torch.manual_seed(self.lozo_random_seed)

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                # Use cached v (don't regenerate to match official behavior)
                v = self.lozo_v[name]

                # Regenerate u with same random seed
                u = torch.randn(param.data.size(0), rank_r,
                               device=param.data.device, dtype=param.data.dtype)

                # Apply weight decay to non-bias, non-layernorm parameters
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - args.learning_rate * (
                        self.projected_grad * (u @ v.t()) + args.weight_decay * param.data)
                else:
                    param.data = param.data - args.learning_rate * (self.projected_grad * (u @ v.t()))
            else:
                # Resample z for bias (same as in perturbation)
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                device=param.data.device, dtype=param.data.dtype)

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - args.learning_rate * (
                        self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - args.learning_rate * (self.projected_grad * z)

    @torch.no_grad()
    def lozo_step(self, model, inputs):
        """
        Estimate gradient by LOZO. Return the loss from f(theta + uv^t).
        Exactly matches official LOZO implementation structure.

        Reference: https://github.com/optsuite/LOZO
        """
        args = self.args

        # Initialize step counter and v dict (matches official: only init once)
        if hasattr(self, 'lozo_step_count'):
            self.lozo_step_count += 1
        else:
            self.lozo_step_count = 0
            self.lozo_v = {}  # Only initialize once, not every step

        # Refresh named_parameters_to_optim (matches official)
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling
        self.lozo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.lozo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.lozo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        # Compute projected gradient
        self.projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()

        # No gradient accumulation support
        assert args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.lozo_perturb_parameters(scaling_factor=1)

        # Update parameters (integrated here instead of separate call)
        self.lozo_update()

        return loss1

    @torch.no_grad()
    def zo_step_v1(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        for i_q in range(args.q):  # TODO shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                if i_q == 0:
                    param.grad = graddiff_times_z / args.q
                else:
                    param.grad += graddiff_times_z / args.q
                # if i_q == args.q - 1:
                #     self.optimizer.step()  # TODO If q > 1, We cannot use this trick anymore. This will cause repeated update.
                #     # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                #     param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q
        self.optimizer.step()
        self.optimizer.zero_grad()

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        # # Optimizer step
        # self.optimizer.step()
        # print(type(self.optimizer), self.optimizer)
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.
        # self.optimizer.zero_grad()
        # model.zero_grad()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                ShardedDDPOption.ZERO_DP_2 in getattr(self.args, 'sharded_ddp', [])
                or ShardedDDPOption.ZERO_DP_3 in getattr(self.args, 'sharded_ddp', [])
                or getattr(self, 'fsdp', None) is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

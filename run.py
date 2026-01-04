import argparse
import os
import random

import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

from metrics import calculate_metric
# from modeling_mistral import (
#     MistralForCausalLM,
#     MistralConfig
# )
from modeling_roberta import (
    RobertaForSequenceClassificationZO,
    RobertaForMultipleChoiceZO,
    load_roberta_for_task
)
from tasks import get_task
from trainer import OurTrainer
from utils import *

os.environ["TRANSFORMERS_CACHE"] = "./cache"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only register custom Mistral if not already in transformers (for older versions)
try:
    AutoConfig.register("mistral", MistralConfig)
    AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)
except ValueError:
    # Already registered in newer transformers versions
    pass


@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP, SNLI, MNLI

    # Number of examples
    num_train: int = 0  # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None  # (only enabled with training) number of development samples
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = 0  # designated seed to sample training samples/demos
    result_file: str = None  # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = False  # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    template_ver: int = 0  # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.

    # RoBERTa MLM mode
    use_roberta_mlm: bool = False  # whether to use MLM-style classification for RoBERTa (template_ver=2 will be used automatically)

    # Training
    trainer: str = "none"
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo_sgd: zeroth-order SGD (MeZO) training
    ## - zo_adam: zeroth-order Adam training
    ## - hizoo: HiZOO (Hessian-Informed Zeroth-Order Optimization)
    ## - lozo: LOZO (Low-rank Zeroth-Order Optimization)
    ## - bszo_v3: BSZO V3 (Bayesian Subspace Zeroth-Order)
    ## - bszo_v4: BSZO V4 (Bayesian Subspace Zeroth-Order)
    optimizer: str = "adamw"
    ## options
    ## - sgd
    ## - adam
    ## - adamw # this is huggingface default
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification
    momentum: float = 0.0  # only work for SGD optimizer

    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO
    perturbation_mode: str = "two_side"
    q: int = 1  # number of Gaussian samples for zeroth-order trainers

    # WandB configuration
    wandb_project: str = "zo-bench"  # WandB project name
    wandb_run_name: str = None  # WandB run name (if None, will use args.tag)

    # BSZO (Bayesian Subspace Zeroth-Order) parameters
    bszo_iter_per_step: int = 1  # number of BSZO iterations per training step
    bszo_fixed_subspace_dim: int = 1  # fixed subspace dimension (2 recommended)

    # Bayesian Subspace Gradient parameters
    bayesian_sigma_prior: str = "auto"  # Prior std: "auto" = √(n/1e6), or specify float value
    bayesian_sigma_noise: str = "auto"  # Noise std: "auto" = √(n/1e6), or specify float value
    bayesian_num_samples: int = 4  # Number of samples per optimization step
    bayesian_adaptive_sampling: bool = True  # Use adaptive sampling (sample in max uncertainty direction)
    bayesian_adaptive_noise: bool = True  # Enable adaptive noise estimation (V4.1)
    bayesian_noise_ema_alpha: float = 0.1  # EMA coefficient for noise estimation
    bayesian_one_sided: bool = False  # Use one-sided finite difference (faster, 1+samples fwd instead of 1+2*samples)

    # HiZOO parameters (Hessian-Informed Zeroth-Order Optimization)
    hizoo_smooth: float = 1e-8  # Hessian smooth coefficient (alpha), default matches official HiZOO
    hizoo_eps: float = 1e-8  # Epsilon for numerical stability in adaptive learning rate

    # LOZO parameters (Low-rank Zeroth-Order Optimization)
    lozo_rank: int = 2  # rank r in LOZO low-rank perturbation (default 2)
    lozo_step_interval: int = 50  # ν in LOZO: how often to resample v matrix (default 50)

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    prefix_init_by_real_act: bool = True  # initialize prefix by real activations of random words

    # prompt tuning hyperparameters
    prompt_tuning: bool = False  # whether to use prompt tuning
    num_virtual_tokens: int = 10  # number of prompt tokens to use
    prompt_init_by_real_tokens: bool = False  # whether to sample random tokens from Embedding layer

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA
    lora_init_strategy: int = 0  # LoRA initialization strategy: 0=standard(zeros), 1=small_normal(recommended for ZO), 2=symmetric, 3=sparse, 4=orthogonal

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Batch-level evaluation (for NEWUOAs-MeZO and other optimizers with batch-level rollback)
    eval_batch: int = None  # evaluate every N batches (if None, use eval_steps instead)

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Early stopping
    early_stopping_patience: int = None  # stop training if eval metric doesn't improve for N evaluations (requires load_best_model_at_end=True)

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = False  # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)

    clean_model_at_end: bool = True  # remove everthing at the end.

    # sparse gradient pruning
    gradient_sparsity: float = None
    sparse_gradient_resample_steps: int = 1
    sparse_gradient_group: str = "layer"
    """
    Options
    ## - global: global sparsity will assign different sparsity to each layer, based on the pretrained weight magnitude
    ## - layer: each layer has the same sparsity
    """

    # module-wise perturbation
    module_wise_perturbation: bool = False
    perturbed_module_level: str = "transformer-block"
    coordinate_perturbation: bool = True  # If True, will update weight right after the gradient is computed
    """
    Options
    ## - transformer-block: perturb one transformer block at a time
    ## - mlp-attn: perturb one mlp/attention layer at a time
    ## - linear: perturb one linear layer at a time
    """


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        """
        Load HuggingFace models
        """
        # Determine precision for logging
        if self.args.load_float16:
            precision_str = "FP16"
        elif self.args.load_bfloat16:
            precision_str = "BF16"
        else:
            precision_str = "FP32"
        with count_time(f"Loading model with {precision_str}"):
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
            print(f"Free GPU memory: {free_in_GB} GB")
            # Reserve memory for CUDA context, activations, etc.
            # For ZO optimizers, 2GB is usually enough
            reserved_memory = min(2, free_in_GB - 2)
            max_memory_per_gpu = max(1, free_in_GB - reserved_memory)  # At least 1GB

            # HiZOO needs extra memory for hizoo_v (Hessian diagonal estimate)
            # which is the same size as model parameters
            if getattr(self.args, 'trainer', None) == 'hizoo':
                # Ensure we have enough total memory across all GPUs
                num_gpus = torch.cuda.device_count()
                # Each GPU gets less memory to force better distribution
                max_memory_per_gpu = max(2, max_memory_per_gpu // 2)
                # Use 'balanced' to distribute model evenly across GPUs
                device_map_strategy = 'balanced'
                print(f"[HiZOO] Detected HiZOO trainer, reserving extra memory for hizoo_v")
                print(f"[HiZOO] Adjusted max memory per GPU: {max_memory_per_gpu} GB x {num_gpus} GPUs")
                print(f"[HiZOO] Using device_map='{device_map_strategy}' for even distribution")
            else:
                device_map_strategy = 'auto'
                print(f"Max memory per GPU: {max_memory_per_gpu} GB")
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                # Head tuning
                if "opt" in self.args.model_name.lower():
                    from modeling_opt import OPTForCausalLM
                    model = OPTForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map=device_map_strategy,
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{max_memory_per_gpu}GB' for i in
                                    range(torch.cuda.device_count())},
                    )
                elif "llama" in self.args.model_name.lower():
                    from modeling_llama import LlamaForCausalLMWithHeadTuning
                    model = LlamaForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map=device_map_strategy,
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{max_memory_per_gpu}GB' for i in
                                    range(torch.cuda.device_count())},
                    )
                elif "mistral" in self.args.model_name.lower():
                    from modeling_mistral import MistralForCausalLMWithHeadTuning
                    model = MistralForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map=device_map_strategy,
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{max_memory_per_gpu}GB' for i in
                                    range(torch.cuda.device_count())},
                    )
                else:
                    raise NotImplementedError(f"Head tuning is not supported for {self.args.model_name}")
            elif "roberta" in self.args.model_name.lower():
                # RoBERTa model (encoder-only, not causal LM)
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16

                # Determine task type and number of labels from task
                task_name = self.args.task_name.lower()

                # Check if using MLM mode
                use_mlm = self.args.use_roberta_mlm
                if use_mlm:
                    # Automatically set template_ver for MLM template based on task
                    # Most tasks use template_ver=2 for MLM, but BoolQ uses 3 (since 2 is already BoolQTemplateV3)
                    if self.args.template_ver == 0:
                        if task_name == "boolq":
                            self.args.template_ver = 3
                            logger.info("Automatically setting template_ver=3 for BoolQ RoBERTa MLM mode")
                        else:
                            self.args.template_ver = 2
                            logger.info("Automatically setting template_ver=2 for RoBERTa MLM mode")

                # Map tasks to classification types
                if task_name in ["copa", "winogrande", "wsc"]:
                    # Multiple choice tasks
                    task_type = "multiple_choice"
                    num_choices = 2
                    logger.info(f"Loading RoBERTa for multiple choice task: {task_name}")
                else:
                    # Classification tasks (SST2, RTE, CB, BoolQ, etc.)
                    if use_mlm:
                        task_type = "mlm"
                        logger.info(f"Loading RoBERTa for MLM-style classification task: {task_name}")
                    else:
                        task_type = "classification"
                        logger.info(f"Loading RoBERTa for classification task: {task_name}")

                    # Determine number of labels based on task
                    if task_name == "sst2":
                        num_labels = 2  # positive/negative
                    elif task_name in ["rte", "boolq"]:
                        num_labels = 2  # entailment/not_entailment or true/false
                    elif task_name in ["cb", "snli", "mnli"]:
                        num_labels = 3  # entailment/neutral/contradiction
                    elif task_name == "trec":
                        num_labels = 6  # TREC 6-class question classification
                    else:
                        num_labels = 2  # default to binary classification

                # Get label word IDs for MLM mode (MeZO-style)
                label_word_ids = None
                if use_mlm:
                    # Load tokenizer early to get label word IDs
                    temp_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

                    # MeZO-style: Get label words from MLMDataset's label_to_word
                    # Check if task has MLM dataset version with label_to_word
                    task_has_mlm_dataset = False
                    try:
                        from tasks import SST2MLMDataset
                        # Create temporary dataset instance to get label_to_word
                        if task_name == "sst2":
                            temp_dataset = SST2MLMDataset()
                            label_word_list = temp_dataset.get_label_word_list(temp_tokenizer)
                            label_word_ids = label_word_list
                            task_has_mlm_dataset = True
                            logger.info(f"Using MeZO-style label words from SST2MLMDataset")
                    except (ImportError, AttributeError):
                        pass

                    # Fallback to hardcoded label words if no MLM dataset available
                    if not task_has_mlm_dataset:
                        logger.warning(f"No MLMDataset found for {task_name}, using hardcoded label words")
                        # Get label words from task template
                        # IMPORTANT: Must use SINGLE-TOKEN words for RoBERTa MLM
                        if task_name == "sst2":
                            label_words = ["bad", "great"]  # Both are single tokens (verified)
                        elif task_name in ["rte", "boolq"]:
                            label_words = ["No", "Yes"]
                        elif task_name in ["cb", "snli", "mnli"]:
                            # 3-class NLI: entailment=Yes, neutral=Maybe, contradiction=No
                            label_words = ["Yes", "Maybe", "No"]
                        elif task_name == "trec":
                            # TREC 6-class: ABBR, ENTY, DESC, HUM, LOC, NUM
                            label_words = ["short", "thing", "what", "who", "where", "how"]
                        else:
                            label_words = ["No", "Yes"]  # default

                        # Tokenize label words (without special tokens)
                        # Verify all label words are single tokens
                        label_word_ids = []
                        for word in label_words:
                            tokens = temp_tokenizer.encode(word, add_special_tokens=False)
                            if len(tokens) != 1:
                                raise ValueError(
                                    f"Label word '{word}' tokenizes to {len(tokens)} tokens: {tokens}. "
                                    f"RoBERTa MLM requires single-token label words! "
                                    f"Tokens: {temp_tokenizer.convert_ids_to_tokens(tokens)}"
                                )
                            label_word_ids.append(tokens[0])

                        logger.info(f"Label words: {label_words}, IDs: {label_word_ids}")

                    logger.info(f"Final label word IDs: {label_word_ids}")
                    logger.info(f"Label tokens: {[temp_tokenizer.convert_ids_to_tokens([id])[0] for id in label_word_ids]}")

                # Load RoBERTa model
                if self.args.no_auto_device:
                    # No auto device (use for FSDP)
                    model = load_roberta_for_task(
                        self.args.model_name,
                        task_type=task_type,
                        num_labels=num_labels if task_type == "classification" else None,
                        num_choices=num_choices if task_type == "multiple_choice" else None,
                        use_mlm=use_mlm,
                        label_word_ids=label_word_ids,
                        torch_dtype=torch_dtype,
                        load_in_8bit=self.args.load_int8,
                    )
                else:
                    # Auto device loading
                    model = load_roberta_for_task(
                        self.args.model_name,
                        task_type=task_type,
                        num_labels=num_labels if task_type == "classification" else None,
                        num_choices=num_choices if task_type == "multiple_choice" else None,
                        use_mlm=use_mlm,
                        label_word_ids=label_word_ids,
                        device_map=device_map_strategy,
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{max_memory_per_gpu}GB' for i in range(torch.cuda.device_count())},
                        load_in_8bit=self.args.load_int8,
                    )
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, )
            else:
                # Auto device loading (for causal LM models)
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, device_map=device_map_strategy,
                                                             torch_dtype=torch_dtype,
                                                             max_memory={i: f'{max_memory_per_gpu}GB' for i in
                                                                         range(torch.cuda.device_count())},
                                                             load_in_8bit=self.args.load_int8, )
            model.eval()

        # Load tokenizer
        #  In mezo, use_fast is set to False. But TypeError will occur when running SQuaD. Setting to be True can fix.
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        if ("llama" in self.args.model_name) or ("mistral" in self.args.model_name.lower()):
            # LLaMA padding token
            tokenizer.pad_token_id = 0  # technically <unk>

        if "roberta" in self.args.model_name.lower():
            # RoBERTa uses <pad> token for padding
            # Ensure proper padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set RoBERTa padding token to: {tokenizer.pad_token}")

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix_tuning import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam,
                         float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16, init_strategy=self.args.lora_init_strategy)

        if self.args.prompt_tuning:
            from prompt_tuning import PromptTuning
            print("Adding Prompt Tuning to model...")
            PromptTuning(
                model,
                num_virtual_tokens=self.args.num_virtual_tokens,
                init_by_real_tokens=self.args.prompt_init_by_real_tokens,
                hide_virtual_token_logits=True,  # a workaround for the other loss/prediction functions
            )
            print("Total/Trainable number of parameters: {}/{}".format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            ))

        if self.args.head_tuning:
            if model.config.model_type in ["opt", "llama", "mistral"]:
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        For RoBERTa, return classification logits.
        This function is only for inference
        """
        # Check if this is a RoBERTa model (encoder-only)
        is_roberta = "roberta" in self.args.model_name.lower()

        if is_roberta:
            # RoBERTa: Direct classification, no need for token-by-token log probs
            # Return logits for each class
            with torch.inference_mode():
                self.model.eval()
                # Get device
                device = next(self.model.parameters()).device

                # Prepare inputs
                if isinstance(input_ids, list):
                    input_ids = torch.tensor([input_ids], device=device)
                elif isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    input_ids = input_ids.to(device)

                # Check if using MLM mode
                use_mlm = self.args.use_roberta_mlm

                if use_mlm:
                    # MLM mode: Extract logits for label words at mask position
                    # Forward pass returns (batch_size, seq_len, vocab_size)
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

                    # Find mask token position
                    mask_token_id = self.tokenizer.mask_token_id
                    mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

                    if len(mask_positions) == 0:
                        # No mask token found - use last position as fallback
                        mask_pos = input_ids.size(1) - 1
                        logger.warning(f"No mask token found, using last position: {mask_pos}")
                    else:
                        mask_pos = mask_positions[0].item()

                    # Extract logits at mask position
                    mask_logits = logits[0, mask_pos, :]  # Shape: (vocab_size,)

                    # Extract only logits for label words
                    if hasattr(self.model, 'label_word_ids') and self.model.label_word_ids is not None:
                        label_word_ids = self.model.label_word_ids
                        class_logits = mask_logits[label_word_ids]  # Shape: (num_labels,)
                    else:
                        logger.error("MLM mode enabled but label_word_ids not set in model")
                        raise RuntimeError("MLM mode requires label_word_ids to be set")

                    return class_logits  # Return 1D tensor of class logits for label words
                else:
                    # Standard classification mode
                    # Forward pass
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits  # Shape: (batch_size, num_labels)

                    # Verify logits shape
                    if logits.dim() == 3:
                        # Wrong shape: (batch_size, seq_len, vocab_size) - might be using MLM model without use_roberta_mlm=True
                        logger.error(f"RoBERTa logits have wrong shape: {logits.shape}")
                        logger.error(f"Expected: (batch_size, num_labels), got: (batch_size, seq_len, vocab_size)")
                        logger.error(f"Model type: {type(self.model)}")
                        logger.error(f"Hint: If using RobertaForMaskedLM, set use_roberta_mlm=True")
                        raise RuntimeError(
                            f"RoBERTa model logits shape mismatch. "
                            f"Logits shape: {logits.shape}, expected: (batch_size, num_labels). "
                            f"Set use_roberta_mlm=True if using MLM-style classification."
                        )

                    # Return logits for this candidate
                    return logits[0]  # Return 1D tensor of class logits

        # Standard causal LM processing
        # Get the device from the model's embedding layer (critical for device_map='auto')
        # When using device_map='auto', different layers may be on different devices
        # We need to ensure input_ids is on the same device as the embedding layer
        try:
            device = None
            # Try to get embedding layer device for OPT models
            # Note: For OPT, the structure is model.model.decoder.embed_tokens
            try:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                    if hasattr(self.model.model.decoder, 'embed_tokens'):
                        device = self.model.model.decoder.embed_tokens.weight.device
            except (AttributeError, RuntimeError):
                pass

            # For LLaMA/Mistral models: model.model.embed_tokens
            if device is None:
                try:
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                        device = self.model.model.embed_tokens.weight.device
                except (AttributeError, RuntimeError):
                    pass

            # Fallback: try to get device attribute
            if device is None:
                try:
                    if hasattr(self.model, 'device'):
                        device = self.model.device
                except (AttributeError, RuntimeError):
                    pass

            # Last resort: get first parameter's device
            if device is None:
                try:
                    device = next(self.model.parameters()).device
                except (StopIteration, RuntimeError):
                    pass

            if device is None:
                raise RuntimeError("Could not determine model device")

        except Exception as e:
            print(f"Warning: Could not determine model device, using cuda:0. Error: {e}")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Convert input_ids to tensor if it's a list, and move to the correct device
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device=device)
        elif isinstance(input_ids, torch.Tensor):
            # Ensure proper shape and device
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            # Always move to correct device (even if already a tensor)
            input_ids = input_ids.to(device)
        else:
            raise TypeError(f"input_ids must be a list or torch.Tensor, got {type(input_ids)}")

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(input_ids, do_sample=args.sampling, temperature=args.temperature,
                                          num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                                          max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                                          num_return_sequences=1,
                                          eos_token_id=[
                                              self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                                              self.tokenizer.eos_token_id], )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]

    def _one_step_pred_roberta(self, train_samples, eval_sample, verbose=False):
        """
        RoBERTa-specific prediction for CLS and MLM classification modes.

        For MLM mode: Encode with <mask> token, get logits at mask position, predict label
        For CLS mode: Encode raw text, get classification logits, predict label
        """
        import torch

        template = self.task.get_template(template_version=self.args.template_ver)

        if self.args.use_roberta_mlm:
            # MLM mode: use encode_prompt_roberta_mlm
            from utils import encode_prompt_roberta_mlm
            encoding, mask_pos, _ = encode_prompt_roberta_mlm(
                self.task, template, train_samples, eval_sample,
                self.tokenizer, max_length=self.args.max_length
            )
        else:
            # CLS mode: use encode_prompt_roberta
            from utils import encode_prompt_roberta
            encoding, _ = encode_prompt_roberta(
                self.task, template, train_samples, eval_sample,
                self.tokenizer, max_length=self.args.max_length
            )
            mask_pos = None

        # Prepare input tensors
        input_ids = torch.tensor([encoding]).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        with torch.no_grad():
            if self.args.use_roberta_mlm:
                # MLM mode: need to manually extract logits at mask position
                # Don't pass labels so we get full prediction_scores
                outputs = self.model.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                sequence_output = outputs[0]  # [batch, seq_len, hidden]
                prediction_scores = self.model.lm_head(sequence_output)  # [batch, seq_len, vocab]

                # Extract logits at mask position
                mask_logits = prediction_scores[0, mask_pos]  # [vocab_size]

                # Get label word IDs and extract only those logits
                label_word_ids = self.model.label_word_ids
                if label_word_ids is not None:
                    label_word_ids_tensor = torch.tensor(label_word_ids, device=mask_logits.device)
                    logits = mask_logits[label_word_ids_tensor]  # [num_classes]
                else:
                    logits = mask_logits

                predicted_label = logits.argmax(dim=-1).item()
            else:
                # CLS mode: standard forward, returns [batch, num_classes]
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                logits = outputs.logits[0]  # [num_classes]
                predicted_label = logits.argmax(dim=-1).item()

        if verbose:
            logger.info(f"[RoBERTa {'MLM' if self.args.use_roberta_mlm else 'CLS'}] Logits: {logits.cpu().tolist()}")
            logger.info(f"[RoBERTa] Predicted label: {predicted_label}")

        # Get correct label
        if hasattr(eval_sample, 'correct_candidate'):
            if isinstance(eval_sample.correct_candidate, list):
                correct_label = eval_sample.candidates.index(eval_sample.correct_candidate[0])
            else:
                correct_label = eval_sample.candidates.index(eval_sample.correct_candidate)
        else:
            correct_label = 0

        return Prediction(correct_candidate=correct_label, predicted_candidate=predicted_label)

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        is_roberta = "roberta" in self.args.model_name.lower()

        # Special handling for RoBERTa models (CLS or MLM classification)
        if is_roberta and self.args.train_as_classification:
            return self._one_step_pred_roberta(train_samples, eval_sample, verbose)

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(self.task,
                                                        self.task.get_template(template_version=self.args.template_ver),
                                                        train_samples, eval_sample,
                                                        self.tokenizer, max_length=self.args.max_length,
                                                        generation=self.task.generation,
                                                        max_new_tokens=self.args.max_new_tokens)

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(
                template_version=self.args.template_ver), train_samples,
                                                                    eval_sample, self.tokenizer,
                                                                    max_length=self.args.max_length, sfc=self.args.sfc,
                                                                    icl_sfc=self.args.icl_sfc,
                                                                    generation=self.task.generation,
                                                                    max_new_tokens=self.args.max_new_tokens)

        outputs = []
        is_roberta = "roberta" in self.args.model_name.lower()

        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            # if verbose:
            #     logger.info("=== Prompt ===")
            #     logger.info(self.tokenizer.decode(encoded_candidates[0]))
            #     logger.info(f"Output: {output_text}")
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        elif is_roberta and len(encoded_candidates) > 1:
            # RoBERTa with multiple candidates: Get logits for each candidate
            all_logits = []
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                logits = self.forward(encoded_candidate)  # Returns class logits
                if verbose and candidate_id == 0:
                    logger.info(f"[RoBERTa Multi-Candidate] Logits shape for candidate 0: {logits.shape}")
                all_logits.append(logits.cpu().detach())

            # Stack all logits and take softmax
            all_logits = torch.stack(all_logits)  # Shape: (num_candidates, num_classes)
            if verbose:
                logger.info(f"[RoBERTa Multi-Candidate] Stacked all_logits shape: {all_logits.shape}")
                logger.info(f"[RoBERTa Multi-Candidate] Expected: (num_candidates={len(encoded_candidates)}, num_classes)")

            # For classification, we want the logit for the correct class for each candidate
            # Each candidate corresponds to a class, so we take the diagonal
            try:
                scores = [all_logits[i, i].item() for i in range(len(all_logits))]
            except (IndexError, RuntimeError) as e:
                logger.error(f"[RoBERTa Multi-Candidate] Error accessing all_logits[i, i]")
                logger.error(f"[RoBERTa Multi-Candidate] all_logits.shape: {all_logits.shape}")
                logger.error(f"[RoBERTa Multi-Candidate] Trying to access indices: {list(range(len(all_logits)))}")
                logger.error(f"[RoBERTa Multi-Candidate] Error: {e}")
                raise

            if verbose:
                logger.info(f"RoBERTa prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    # if candidate_id == 0:
                    #     logger.info("=== Candidate %d ===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate))
                    # else:
                    #     logger.info("=== Candidate %d (without context)===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id],
                                                          option_len=sfc_option_lens[
                                                              candidate_id])  # if verbose:  #     logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)  #     logger.info(  #         self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])  #     logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs,
                                "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False, description=None):
        """
        Evaluate function.
        Here, train_samples are used for demonstrations for ICL.
        If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        Otherwise, the same training set is used for all eval samples.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(eval_samples):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                                   eval_sample, verbose=False))

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self, train_samples, dev_samples, eval_samples):
        """
        Training function
        if self.num_dev is not None, eval_samples are dev_samples
        """
        logger.info(f"Eval sample length is {len(eval_samples)}")

        # Set tokenizer padding side based on model type
        if "roberta" in self.args.model_name.lower():
            # RoBERTa uses right padding (encoder models)
            self.tokenizer.padding_side = "right"
            logger.info("Using right padding for RoBERTa")
        else:
            # Causal LM models use left padding (so that all the options are right aligned)
            self.tokenizer.padding_side = "left"
            logger.info("Using left padding for causal LM")

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []

            # Check if using RoBERTa (encoder-only model)
            is_roberta = "roberta" in self.args.model_name.lower()

            for sample in samples:
                if is_roberta and self.args.train_as_classification:
                    # RoBERTa: Use special encoding that returns single input + label
                    if self.args.use_roberta_mlm:
                        # MLM-style classification
                        from utils import encode_prompt_roberta_mlm

                        # For RoBERTa MLM, encode with mask token and get mask position
                        encoding, mask_pos, label_id = encode_prompt_roberta_mlm(
                            self.task,
                            self.task.get_template(template_version=self.args.template_ver),
                            [],
                            sample,
                            self.tokenizer,
                            max_length=self.args.max_length
                        )

                        # RoBERTa MLM: labels are class indices, with mask position
                        data.append({
                            "input_ids": encoding,
                            "labels": label_id,
                            "mask_pos": mask_pos,
                            "attention_mask": [1] * len(encoding)
                        })
                    else:
                        # Standard CLS-based classification
                        from utils import encode_prompt_roberta

                        # For RoBERTa classification, encode once per sample (not per candidate)
                        encoding, label_id = encode_prompt_roberta(
                            self.task,
                            self.task.get_template(template_version=self.args.template_ver),
                            [],
                            sample,
                            self.tokenizer,
                            max_length=self.args.max_length
                        )

                        # RoBERTa classification: labels are class indices
                        data.append({
                            "input_ids": encoding,
                            "labels": label_id,
                            "attention_mask": [1] * len(encoding)
                        })
                    continue

                # Standard processing for causal LM models
                encoded_candidates, option_lens = encode_prompt(self.task, self.task.get_template(
                    template_version=self.args.template_ver), [], sample,
                                                                self.tokenizer, max_length=self.args.max_length,
                                                                generation=self.task.generation,
                                                                generation_with_gold=True,
                                                                max_new_tokens=self.args.max_new_tokens)
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][
                                                               :-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id,
                                  "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in
                                 range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
            dev_dataset = HFDataset(_convert(dev_samples))

        # Wrap forward function based on model type
        is_roberta = "roberta" in self.args.model_name.lower()

        if is_roberta and self.args.train_as_classification:
            # For RoBERTa, use special forward wrapper
            logger.info("Wrapping RoBERTa forward function for classification")
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len_roberta.__get__(self.model, type(self.model))
        elif self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            # This is for causal LM models
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.gradient_sparsity is not None:
            logger.info(
                f"[Sparse gradient] sparsity is {self.args.gradient_sparsity}, resampling per {self.args.sparse_gradient_resample_steps} steps"
            )

            if self.args.sparse_gradient_group == "global":
                logger.info(f"[Sparse gradient] global-ratio random pruning is enabled, "
                            f"sparsity of each layer is computed based on the pretrained weight magnitude.")
            elif self.args.sparse_gradient_group == "layer":
                logger.info(f"[Sparse gradient] layer-wise random pruning is enabled, "
                            f"sparsity of each layer is the same.")
            else:
                raise NotImplementedError(f"Unknown sparse gradient group: {self.args.sparse_gradient_group}")

        perturb_module_regex = None
        if self.args.module_wise_perturbation:
            if "opt" in self.args.model_name:
                assert self.args.perturbed_module_level in OPT_PERTURBATION_LEVEL_TO_REGEX.keys(), f"Unknown perturbed module group {self.args.perturbed_module_level}"
                perturb_module_regex = OPT_PERTURBATION_LEVEL_TO_REGEX[self.args.perturbed_module_level]
            else:
                raise NotImplementedError(f"Unimplemented model {self.args.model_name} for module-wise perturbation")

        trainer = OurTrainer(model=self.model,
                             args=self.args,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             tokenizer=self.tokenizer,
                             data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                                             pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                                 self.tokenizer, pad_to_multiple_of=8),
                             eval_samples=eval_samples,
                             dev_samples=dev_samples,
                             evaluate_func=self.evaluate,
                             perturb_module_regex=perturb_module_regex,
                             framework=self,
                             )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Early stopping callback
        if self.args.early_stopping_patience is not None:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=self.args.early_stopping_patience))

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # This calls the trainer._inner_training_loop()
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save trainer reference for accessing best_test_acc later
        self.trainer = trainer

        # Explicitly save the model
        if self.args.save_model:
            logger.info("Save model..")
            trainer.save_model()

        # FSDP compatibility
        self.model = trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

    def delete_checkpoints(self):
        import shutil
        print(f"\nWARNING: Removing everything at end: {self.args.output_dir}")
        deleted_folders = [folder for folder in os.listdir(self.args.output_dir)
                           if os.path.isdir(os.path.join(self.args.output_dir, folder))
                           and folder.startswith("checkpoint-")]
        for f in deleted_folders:
            shutil.rmtree(os.path.join(self.args.output_dir, f))
        print(f"deleted folders: ", deleted_folders)


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()
    if args.prefix_tuning:
        args.mode = "prefix"
    elif args.lora:
        args.mode = "lora"
    elif args.prompt_tuning:
        args.mode = "prompt"
    else:
        args.mode = "ft"
    args.tag = f"{args.trainer}-{args.task_name}-{args.template_ver}-{args.model_name.split('/')[-1]}-OPTIM_{args.mode}-STEP{args.max_steps}-{args.optimizer}-LR{args.learning_rate}-{args.lr_scheduler_type}-ZOEPS{args.zo_eps}-Q{args.q}"
    args.tag = "momen" + args.tag if args.momentum > 0 else args.tag
    args.tag = f"sparse_grad-{args.gradient_sparsity}-{args.sparse_gradient_group}-{args.sparse_gradient_resample_steps}-" + args.tag if args.gradient_sparsity is not None else args.tag
    args.tag = f"module_perturb-{args.perturbed_module_level}-" + args.tag if args.module_wise_perturbation else args.tag
    args.run_name = args.tag
    # 只有当用户没有指定自定义 output_dir 时才使用自动生成的路径
    # 默认的 output_dir 通常是 "result" 或以 "result/" 开头但没有具体子目录
    if args.output_dir in ["result", "result/", None, ""] or args.output_dir == "output":
        args.output_dir = f"result/{args.tag}"
    args.result_file = os.path.join(args.output_dir, "results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    args.logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.logging_dir, exist_ok=True)

    # Initialize WandB with custom project and run name
    wandb_project = os.getenv("WANDB_PROJECT", args.wandb_project)
    wandb_run_name = os.getenv("WANDB_NAME", args.wandb_run_name if args.wandb_run_name else args.tag)
    wandb.init(project=wandb_project, name=wandb_run_name, config=args)

    set_seed(args.seed)
    task = get_task(args.task_name)

    # This function samples both training and validation samples. The validation (dev) samples are also stored in "train_sets"
    # Later the train_samples and dev_samples are separated
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval,
                                        num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    framework = Framework(args, task)

    # ZO-Bench Added
    # We add these parameters to evaluate the model during the training.
    # These two parameters will be used in the training loop
    # args.task = task
    # args.framework = framework

    if args.train_set_seed is not None or args.num_train_sets is not None:

        # Training goes to this way

        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                # Here the training samples are seperated
                if args.num_dev is not None:
                    # Dev samples
                    # assert args.num_dev + args.num_train <= len(train_samples), f"num_dev({args.num_dev})+num_train({args.num_train}) is more than actual num of training samples ({len(train_samples)})."
                    dev_samples = train_samples[-args.num_dev:]
                    train_samples = train_samples[:-args.num_dev]
                    logger.info("Dev samples: %d" % len(dev_samples))
                    logger.info("Train samples: %d" % len(train_samples))
                else:
                    dev_samples = None
                    logger.info("Train samples: %d" % len(train_samples))
                    logger.info("No dev samples")

                args.dev_samples = dev_samples
                args.eval_samples = eval_samples

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples, eval_samples)

                if not args.no_eval:  # This is True
                    metrics = framework.evaluate([], eval_samples, description="Evaluating on the Test Set")
                    _keys = list(metrics.keys())
                    for m in _keys:
                        metrics["test_" + m] = metrics[m]
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate(
                            [], dev_samples, description="Evaluating on the Validation Set"
                        )
                        _keys = list(dev_metrics.keys())
                        for m in _keys:
                            metrics["val_" + m] = dev_metrics[m]

                    # Log final test accuracy as a single summary metric (for best model)
                    # Use best_test_acc from trainer if available (recorded during training when best val_acc was found)
                    # This avoids issues with load_best_model_at_end failing for sharded safetensors
                    best_test_acc = getattr(framework.trainer, 'best_test_acc', None) if hasattr(framework, 'trainer') else None
                    if best_test_acc is not None:
                        wandb.run.summary["final_test_accuracy"] = best_test_acc
                        wandb.log({"final_test_accuracy": best_test_acc})
                        logger.info(f"Final test accuracy (best model): {best_test_acc:.4f}")
                    elif "accuracy" in metrics:
                        # Fallback to current model's accuracy if best_test_acc not available
                        wandb.run.summary["final_test_accuracy"] = metrics["accuracy"]
                        wandb.log({"final_test_accuracy": metrics["accuracy"]})
                        logger.info(f"Final test accuracy (current model): {metrics['accuracy']:.4f}")
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)
            logger.info(metrics)
            wandb.log(metrics)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                wandb.log(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" + result_file_tag(
                        args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
            if args.trainer != "none" and args.clean_model_at_end:
                framework.delete_checkpoints()

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples
        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        wandb.log(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(
                args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)


if __name__ == "__main__":
    main()

import contextlib
import json
import logging
import signal
import time
from collections.abc import Mapping
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, List, NewType, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import PaddingStrategy

InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# fixme: only support OPT now
OPT_PERTURBATION_LEVEL_TO_REGEX = {
    "transformer-block": r".*model\.decoder\.layers\.[0-9]+$"
                         r"|.*model\.decoder\.embed_.*"
                         r"|.*model\.decoder\.final_layer_norm$",
    "mlp-attn": r".*model\.decoder\.layers\.[0-9]+\.self_attn$"
                r"|.*model\.decoder\.layers\.[0-9]+\.fc[12]$"
                r"|.*model\.decoder\.layers\.[0-9]+\.final_layer_norm$"
                r"|.*model\.decoder\.layers\.[0-9]+\.self_attn_layer_norm$"
                r"|.*model\.decoder\.embed_.*"
                r"|.*model\.decoder\.final_layer_norm$",
    "linear": r".*model\.decoder\.layers\.[0-9]+\.self_attn.[qkv]_proj$"
              r"|.*model\.decoder\.layers\.[0-9]+\.self_attn.out_proj$"
              r"|.*model\.decoder\.layers\.[0-9]+\.fc[12]$"
              r"|.*model\.decoder\.layers\.[0-9]+\.final_layer_norm$"
              r"|.*model\.decoder\.layers\.[0-9]+\.self_attn_layer_norm$"
              r"|.*model\.decoder\.embed_.*"
              r"|.*model\.decoder\.final_layer_norm$",
}


def forward_wrap_with_option_len(
        self,
        input_ids=None,
        labels=None,
        option_len=None,
        num_options=None,
        return_dict=None,
        **kwargs
):
    """
    This is to replace the original forward function of Transformer models to enable:
    (1) Partial target sequence: loss will only be calculated on part of the sequence
    (2) Classification-style training: a classification loss (CE) will be calculated over several options
    Input:
    - input_ids, labels: same as the original forward function
    - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
      last option_len tokens
    - num_options: a list of int indicating the number of options for each example (this will be #label
      words for classification tasks and #choices for multiple choice tasks), and a classification loss
      will be calculated.
    """
    # Get the device from the model's embedding layer (critical for device_map='auto')
    # When using device_map='auto', different layers may be on different devices
    # We need to ensure input_ids is on the same device as the embedding layer
    if input_ids is not None and isinstance(input_ids, torch.Tensor):
        try:
            device = None
            # Try to get embedding layer device for OPT models
            try:
                if hasattr(self, 'model') and hasattr(self.model, 'decoder'):
                    if hasattr(self.model.decoder, 'embed_tokens'):
                        device = self.model.decoder.embed_tokens.weight.device
            except (AttributeError, RuntimeError):
                pass

            # For LLaMA/Mistral models
            if device is None:
                try:
                    if hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
                        device = self.model.embed_tokens.weight.device
                except (AttributeError, RuntimeError):
                    pass

            # Fallback: try to get device attribute
            if device is None:
                try:
                    if hasattr(self, 'device'):
                        device = self.device
                except (AttributeError, RuntimeError):
                    pass

            # Last resort: get first parameter's device
            if device is None:
                try:
                    device = next(self.parameters()).device
                except (StopIteration, RuntimeError):
                    pass

            # Move input_ids to the correct device if needed
            if device is not None and input_ids.device != device:
                input_ids = input_ids.to(device)
        except Exception as e:
            # If we can't determine the device, try to infer from input_ids and hope for the best
            pass

    # Remove input_ids from kwargs if present to avoid conflict
    kwargs.pop('input_ids', None)

    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs

    # in prompt tuning, we need to remove the virtual tokens from the logits to match the input ids
    logits = outputs.logits

    loss = None
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # Apply option len (do not calculate loss on the non-option part)
    # for _i, _len in enumerate(option_len):
    #     shift_labels[_i, :-_len] = -100
    # re-write the above code to avoid the for loop
    non_option_len = shift_labels.shape[1] - option_len
    mask = torch.arange(
        shift_labels.shape[1], device=shift_labels.device
    ).expand(shift_labels.shape[0], -1) < non_option_len.unsqueeze(-1)
    shift_labels[mask] = -100

    # Calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None:
        # Train as a classification tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100  # Option part
        shift_labels[~mask] = 0  # So that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(
            -1)  # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1)  # (bsz x num_options)

        if any([x != num_options[0] for x in num_options]):
            # Multi choice tasks with different number of options
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0)  # (1, num_options)
                _labels = labels[start_id:end_id][0].unsqueeze(0)  # (1)
                loss = loss_fct(_logits, _labels) + loss
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            num_options = num_options[0]
            selected_log_probs = selected_log_probs.view(-1, num_options)  # (bsz, num_options)
            labels = labels.view(-1, num_options)[:, 0]  # Labels repeat so we only take the first one
            loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_wrap_with_option_len_roberta(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        mask_pos=None,  # Add explicit mask_pos parameter
        option_len=None,
        num_options=None,
        return_dict=None,
        **kwargs
):
    """
    Forward wrapper for RoBERTa models (encoder-only architecture).

    Key differences from causal LM:
    - Uses [CLS] token for classification, not next-token prediction
    - No autoregressive loss computation
    - Direct classification over label space

    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: For classification tasks, this is the label ID (not token sequences)
        mask_pos: Position of mask token (for MLM-style classification)
        option_len: Not used for RoBERTa (kept for API compatibility)
        num_options: For multiple-choice, indicates number of options per example

    Returns:
        Model output with loss and logits
    """
    # For RoBERTa, we directly pass through to the model
    # Labels should be label IDs, not token sequences
    # IMPORTANT: Pass mask_pos for MLM-style classification
    outputs = self.original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        mask_pos=mask_pos,  # Critical for MLM-style classification
        return_dict=True,
        **kwargs
    )

    return outputs


def encode_prompt_roberta(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False):
    """
    Encode prompts for RoBERTa (encoder-only) models in CLS classification mode.

    Unlike causal LM models, RoBERTa:
    - Uses [CLS] and [SEP] tokens for sequence classification
    - Does not need separate encodings for each candidate
    - Returns a single encoding with the label ID

    For CLS classification, we use encode_for_cls() which returns the raw input
    without any causal LM prompts like "It was".

    Returns:
        - encoding: Single tokenized sequence
        - label_id: Integer label for classification
    """
    # For RoBERTa CLS classification, use encode_for_cls if available
    # This returns raw input without causal LM prompt suffixes
    if hasattr(template, 'encode_for_cls'):
        encode_fn = template.encode_for_cls
    elif sfc or icl_sfc:
        encode_fn = template.encode_sfc if hasattr(template, 'encode_sfc') else template.encode
    else:
        encode_fn = template.encode

    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')

    # For RoBERTa CLS, we typically don't use ICL demonstrations
    # as the model uses [CLS] token for classification directly
    # Skip ICL for CLS mode
    final_prompt = unverbalized_eval_prompt

    # Tokenize
    encoding = tokenizer.encode(final_prompt, max_length=max_length, truncation=True)

    # Get label ID
    if hasattr(eval_sample, 'correct_candidate'):
        if isinstance(eval_sample.correct_candidate, list):
            label_id = eval_sample.candidates.index(eval_sample.correct_candidate[0])
        else:
            label_id = eval_sample.candidates.index(eval_sample.correct_candidate)
    else:
        label_id = 0

    return encoding, label_id


def encode_prompt_roberta_mlm(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False):
    """
    Encode prompts for RoBERTa MLM-style classification.

    This function:
    - Uses templates with <mask> tokens
    - Finds the position of <mask> token in the sequence
    - Returns encoding, mask position, and label ID

    Returns:
        - encoding: Tokenized sequence with <mask> token
        - mask_pos: Position of <mask> token in the sequence
        - label_id: Integer label for classification
    """
    # For RoBERTa MLM, we use mask token in the prompt
    # Demonstrations for ICL (also with mask tokens)
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()

    # Get the input text with mask token
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc if hasattr(template, 'encode_sfc') else template.encode
    else:
        encode_fn = template.encode

    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')

    # Replace template's placeholder mask token with tokenizer's actual mask token
    # SST2TemplateMLM uses "<mask>" as placeholder
    mask_token_str = tokenizer.mask_token  # e.g., "<mask>" for RoBERTa
    unverbalized_eval_prompt = unverbalized_eval_prompt.replace("<mask>", mask_token_str)

    # Combine with demonstrations
    if train_prompts:
        train_prompts = train_prompts.replace("<mask>", mask_token_str)
        final_prompt = (train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')
    else:
        final_prompt = unverbalized_eval_prompt

    # Tokenize
    encoding = tokenizer.encode(final_prompt, max_length=max_length, truncation=True)

    # Find mask token position
    mask_token_id = tokenizer.mask_token_id
    try:
        mask_pos = encoding.index(mask_token_id)
    except ValueError:
        # If mask token not found (e.g., truncated), use last position
        logger.warning(f"Mask token not found in encoding. Using last position.")
        mask_pos = len(encoding) - 1

    # Get label ID
    if hasattr(eval_sample, 'correct_candidate'):
        if isinstance(eval_sample.correct_candidate, list):
            label_id = eval_sample.candidates.index(eval_sample.correct_candidate[0])
        else:
            label_id = eval_sample.candidates.index(eval_sample.correct_candidate)
    else:
        label_id = 0

    return encoding, mask_pos, label_id


def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False,
                  generation=False, generation_with_gold=False, max_new_tokens=None):
    """
    Encode prompts for eval_sample
    Input: 
    - task, template: task and template class
    - train_samples, eval_sample: demonstrations and the actual sample
    - tokenizer, max_length: tokenizer and max length
    - sfc: generate prompts for calibration (surface form competition; https://arxiv.org/abs/2104.08315)
    - icl_sfc: generate prompts for ICL version calibration
    - generation: whether it is an generation task
    - generation_with_gold: whether to include the generation-task gold answers (for training)
    - max_new_tokens: max number of new tokens to generate so that we can save enough space 
      (only for generation tasks)
    Output:
    - encodings: a list of N lists of tokens. N is the number of options for classification/multiple-choice.
    - option_lens: a list of N integers indicating the number of option tokens.
    """

    # Demonstrations for ICL
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    train_prompts = task.train_sep.join(train_prompts).strip()

    # sfc or icl_sfc indicates that this example is used for calibration
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc
        verbalize_fn = template.verbalize_sfc
    else:
        encode_fn = template.encode
        verbalize_fn = template.verbalize

    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        # We generate one prompt for each candidate (different classes in classification)
        # or different choices in multiple-choice tasks
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for
                       verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc:
            # Without demonstrations
            final_prompts = verbalized_eval_prompts
        else:
            # With demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in
                             verbalized_eval_prompts]
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for
                           verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in
                             verbalized_eval_prompts]
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # Tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    # Truncate (left truncate as demonstrations are less important)
    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if hasattr(tokenizer, 'add_bos_token') and tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length - 1):] for encoding in encodings]
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]

    return encodings, option_lens


@dataclass
class ICLCollator:
    """
    Collator for ICL
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0,
                   "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in
                                enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature

        return batch


@dataclass
class DataCollatorWithPaddingAndNesting:
    """
    Collator for training
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Check if features are nested (list of lists) or flat (list of dicts)
        # For RoBERTa classification, features are already flat dicts
        # For causal LM multi-choice, features are nested lists
        if features and isinstance(features[0], list):
            # Nested structure: flatten it
            features = [ff for f in features for ff in f]
        # Otherwise, features are already flat dicts, no need to flatten

        # Extract non-tokenization fields before padding
        # tokenizer.pad only handles input_ids, attention_mask, token_type_ids, etc.
        import torch

        labels = None
        option_len = None
        num_options = None
        mask_pos = None

        if features and "labels" in features[0]:
            # Check if labels are integers (classification) or sequences (language modeling)
            first_label = features[0]["labels"]
            if isinstance(first_label, int):
                # Classification labels: extract and convert to tensor
                labels = torch.tensor([f["labels"] for f in features])
            # Otherwise, labels are sequences and will be handled by tokenizer.pad

        if features and "option_len" in features[0]:
            option_len = torch.tensor([f["option_len"] for f in features])

        if features and "num_options" in features[0]:
            num_options = torch.tensor([f["num_options"] for f in features])

        if features and "mask_pos" in features[0]:
            # For MLM-style tasks, extract mask positions
            mask_pos = torch.tensor([f["mask_pos"] for f in features])

        # Remove extracted fields from features before padding
        fields_to_remove = set()
        if labels is not None:
            fields_to_remove.add("labels")
        if option_len is not None:
            fields_to_remove.add("option_len")
        if num_options is not None:
            fields_to_remove.add("num_options")
        if mask_pos is not None:
            fields_to_remove.add("mask_pos")

        if fields_to_remove:
            features = [{k: v for k, v in f.items() if k not in fields_to_remove} for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Add extracted fields back to batch
        if labels is not None:
            batch["labels"] = labels
        if option_len is not None:
            batch["option_len"] = option_len
        if num_options is not None:
            batch["num_options"] = num_options
        if mask_pos is not None:
            batch["mask_pos"] = mask_pos

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class MLMDataCollatorWithPadding(DataCollatorMixin):
    """
    MeZO-style data collator for RoBERTa MLM tasks with OurInputFeatures.

    This collator handles:
    - Dynamic padding for input_ids, attention_mask, token_type_ids
    - Proper handling of mask_pos (List[int])
    - Label word lists
    - SFC (Surface Form Calibration) inputs
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        # Separate mask_pos and SFC fields from standard features
        mask_pos = [feature.mask_pos for feature in features] if hasattr(features[0], 'mask_pos') and features[0].mask_pos is not None else None
        labels = [feature.label for feature in features] if hasattr(features[0], 'label') and features[0].label is not None else None
        sfc_input_ids = [feature.sfc_input_ids for feature in features] if hasattr(features[0], 'sfc_input_ids') and features[0].sfc_input_ids is not None else None

        # Extract standard features for padding
        standard_features = []
        for feature in features:
            standard_item = {}
            for field in ["input_ids", "attention_mask", "token_type_ids"]:
                if hasattr(feature, field) and getattr(feature, field) is not None:
                    standard_item[field] = getattr(feature, field)
            standard_features.append(standard_item)

        # Pad standard features
        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Add mask_pos (no padding needed, just convert to tensor)
        if mask_pos is not None:
            # mask_pos is List[List[int]] from tokenize_multipart_input
            # e.g., [[5], [5], [5]] for batch_size=3
            # Keep as list of lists for forward compatibility
            batch["mask_pos"] = mask_pos

        # Add labels
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        # Add SFC inputs if present
        if sfc_input_ids is not None:
            sfc_features = []
            sfc_mask_pos = []
            for feature in features:
                if hasattr(feature, 'sfc_input_ids') and feature.sfc_input_ids is not None:
                    sfc_item = {
                        "input_ids": feature.sfc_input_ids,
                        "attention_mask": feature.sfc_attention_mask,
                    }
                    sfc_features.append(sfc_item)
                    sfc_mask_pos.append(feature.sfc_mask_pos)

            if sfc_features:
                sfc_batch = self.tokenizer.pad(
                    sfc_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                batch["sfc_input_ids"] = sfc_batch["input_ids"]
                batch["sfc_attention_mask"] = sfc_batch["attention_mask"]
                batch["sfc_mask_pos"] = sfc_mask_pos

        return batch


@dataclass
class NondiffCollator(DataCollatorMixin):
    """
    Collator for non-differentiable objectives
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != "gold"} for feature in
                              features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "gold" in features[0]:
            batch["gold"] = [feature["gold"] for feature in features]

        return batch


class SIGUSR1Callback(transformers.TrainerCallback):
    """
    This callback is used to save the model when a SIGUSR1 signal is received
    (SLURM stop signal or a keyboard interruption signal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        logger.warn("Handler registered")

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warn("Signal received")

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), cls=EnhancedJSONEncoder, indent=4)

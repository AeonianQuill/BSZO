"""
Custom RoBERTa model for sequence classification with ZO optimization support.

RoBERTa is an encoder-only model (not causal LM), so it requires different handling
compared to GPT-style models. This module provides:
1. RoBERTaForSequenceClassification with ZO-compatible forward pass
2. RoBERTaForMaskedLM for MLM-style classification
3. Support for classification tasks (SST2, RTE, CB, etc.)
4. Compatible with LoRA, prefix tuning, and prompt tuning
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaForMaskedLM, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
from typing import Optional, Tuple, Union, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RobertaForSequenceClassificationZO(RobertaForSequenceClassification):
    """
    RoBERTa model for sequence classification with ZO optimization support.

    Key differences from causal LM models:
    - Uses [CLS] token representation for classification
    - No autoregressive generation
    - Bidirectional attention (not causal masking)
    - Classification head outputs logits directly
    """

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        logger.info(f"Initializing RoBERTa for sequence classification with {config.num_labels} labels")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ZO-specific parameters (for API compatibility with MLM model)
        option_len: Optional[torch.Tensor] = None,
        num_options: Optional[torch.Tensor] = None,
        mask_pos: Optional[torch.LongTensor] = None,  # Not used in CLS mode, for API compatibility
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for RoBERTa sequence classification.

        Args:
            option_len: Not used for RoBERTa (kept for API compatibility)
            num_options: For multi-choice tasks, indicates number of options per example
            mask_pos: Not used in CLS mode (kept for API compatibility with MLM model)

        Returns:
            SequenceClassifierOutput with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard RoBERTa forward pass
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get [CLS] token representation
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Handle different label types
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceZO(nn.Module):
    """
    RoBERTa model for multiple choice tasks (e.g., Copa, WinoGrande).

    For multiple choice, we need to:
    1. Process each choice independently
    2. Get logits for each choice
    3. Select the choice with highest logit
    """

    def __init__(self, model_name: str, num_choices: int = 2):
        super().__init__()
        from transformers import AutoModel

        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.num_choices = num_choices

        logger.info(f"Initializing RoBERTa for multiple choice with {num_choices} choices")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Forward pass for multiple choice.

        Input shape: (batch_size * num_choices, sequence_length)
        Output shape: (batch_size, num_choices)
        """
        # Get outputs from RoBERTa
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get [CLS] representation
        pooled_output = outputs[1]  # [batch_size * num_choices, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size * num_choices, 1]

        # Reshape to separate choices
        reshaped_logits = logits.view(-1, self.num_choices)  # [batch_size, num_choices]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
        )


class RobertaForMaskedLMZO(RobertaForMaskedLM):
    """
    RoBERTa model for Masked Language Modeling with ZO optimization support.

    This model performs classification by:
    1. Inserting a <mask> token in the input
    2. Predicting label words at the mask position
    3. Computing loss over the label word vocabulary

    This is the proper way to use RoBERTa for few-shot learning, as it leverages
    the bidirectional context and pre-trained MLM objective.
    """

    def __init__(self, config: RobertaConfig, label_word_ids: Optional[List[int]] = None):
        super().__init__(config)
        self.label_word_ids = label_word_ids  # Token IDs for label words (e.g., ["terrible", "great"])

        # Cache mask_token_id to avoid loading tokenizer every forward pass
        self.mask_token_id = 50264  # RoBERTa standard mask token ID

        logger.info(f"Initializing RoBERTa for MLM-style classification")
        logger.info(f"Cached mask_token_id: {self.mask_token_id}")
        if label_word_ids:
            logger.info(f"Label word IDs: {label_word_ids}")

    def set_label_word_ids(self, label_word_ids: List[int]):
        """Set label word token IDs after initialization."""
        self.label_word_ids = label_word_ids
        logger.info(f"Set label word IDs: {label_word_ids}")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ZO-specific parameters
        mask_pos: Optional[torch.LongTensor] = None,  # Position of <mask> token
        option_len: Optional[torch.Tensor] = None,  # For compatibility
        num_options: Optional[torch.Tensor] = None,  # For compatibility
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        Forward pass for RoBERTa MLM-style classification.

        Args:
            input_ids: Input token IDs with <mask> token inserted
            attention_mask: Attention mask
            mask_pos: Position(s) of <mask> token in each sequence
            labels: True label indices (0, 1, ..., num_labels-1)

        Returns:
            MaskedLMOutput with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard RoBERTa MLM forward pass
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # Get prediction scores for all vocabulary
        prediction_scores = self.lm_head(sequence_output)  # [batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # MeZO-style: Prioritize using mask_pos if provided
            if mask_pos is not None:
                # mask_pos from MeZO tokenize_multipart_input: List[List[int]]
                # e.g., [[5], [5], [5]] for batch_size=3
                # Convert to flat tensor: [5, 5, 5]
                if isinstance(mask_pos, list):
                    if isinstance(mask_pos[0], list):
                        # MeZO format: [[pos], [pos], ...] -> flatten to [pos, pos, ...]
                        mask_pos_flat = [pos[0] for pos in mask_pos]
                        mask_pos = torch.tensor(mask_pos_flat, device=input_ids.device)
                    else:
                        # Simple list format: [pos, pos, ...] -> convert to tensor
                        mask_pos = torch.tensor(mask_pos, device=input_ids.device)
                elif not isinstance(mask_pos, torch.Tensor):
                    # Single value or other format
                    mask_pos = torch.tensor([mask_pos], device=input_ids.device)
            else:
                # Fallback: Automatically find mask token positions
                # Use cached mask_token_id (initialized in __init__)
                mask_token_id = self.mask_token_id

                # Find first occurrence of mask token in each sequence
                mask_positions = []
                for i in range(input_ids.size(0)):
                    mask_indices = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
                    if len(mask_indices) > 0:
                        mask_positions.append(mask_indices[0].item())
                    else:
                        # No mask token found, use last position
                        mask_positions.append(input_ids.size(1) - 1)
                        logger.warning(f"No mask token ({mask_token_id}) found in sequence {i}, using last position")

                mask_pos = torch.tensor(mask_positions, device=input_ids.device)

            # Extract logits at mask positions
            batch_size = input_ids.size(0)

            # Gather logits at mask positions: [batch_size, vocab_size]
            mask_logits = prediction_scores[torch.arange(batch_size, device=prediction_scores.device), mask_pos]

            if self.label_word_ids is not None:
                # Classification mode: only consider label word logits
                # Convert label_word_ids to tensor if needed
                if not isinstance(self.label_word_ids, torch.Tensor):
                    label_word_ids_tensor = torch.tensor(
                        self.label_word_ids,
                        device=mask_logits.device,
                        dtype=torch.long
                    )
                else:
                    label_word_ids_tensor = self.label_word_ids.to(mask_logits.device)

                # Extract logits for label words only: [batch_size, num_labels]
                label_logits = mask_logits[:, label_word_ids_tensor]

                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(label_logits, labels.view(-1))

                # For evaluation, we want to return the label logits
                # Store in the output so we can use them for prediction
                prediction_scores = label_logits
            else:
                # Standard MLM mode: predict the actual token at mask position
                # This is for when we have token IDs as labels
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(mask_logits, labels.view(-1))
                prediction_scores = mask_logits

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_roberta_for_task(
    model_name: str,
    task_type: str = "classification",
    num_labels: int = 2,
    use_mlm: bool = False,
    label_word_ids: Optional[List[int]] = None,
    **kwargs
):
    """
    Load appropriate RoBERTa model based on task type.

    Args:
        model_name: HuggingFace model name (e.g., "roberta-large")
        task_type: "classification", "multiple_choice", or "mlm"
        num_labels: Number of labels for classification
        use_mlm: Whether to use MLM-style classification (mask token prediction)
        label_word_ids: Token IDs for label words (for MLM mode)

    Returns:
        RoBERTa model instance
    """
    if task_type == "mlm" or use_mlm:
        # MLM-style classification
        config = RobertaConfig.from_pretrained(model_name)

        # Remove device_map and max_memory
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        kwargs.pop("num_choices", None)

        model = RobertaForMaskedLMZO.from_pretrained(
            model_name,
            config=config,
            label_word_ids=label_word_ids,
            **kwargs
        )

        # Manually move model to GPU if device_map was specified
        if device_map == "auto":
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda:0"))
                logger.info(f"Loaded RoBERTa for MLM-style classification on cuda:0")
            else:
                logger.info(f"Loaded RoBERTa for MLM-style classification on CPU")
        else:
            logger.info(f"Loaded RoBERTa for MLM-style classification")

    elif task_type == "classification":
        config = RobertaConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        # Remove num_choices from kwargs as it's not needed for classification
        kwargs.pop("num_choices", None)

        # Remove device_map and max_memory for RoBERTa classification
        # RoBERTa's classification head is randomly initialized, which causes issues with device_map='auto'
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        model = RobertaForSequenceClassificationZO.from_pretrained(model_name, config=config, **kwargs)

        # Manually move model to GPU if device_map was specified
        if device_map == "auto":
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda:0"))
                logger.info(f"Loaded RoBERTa for classification with {num_labels} labels on cuda:0")
            else:
                logger.info(f"Loaded RoBERTa for classification with {num_labels} labels on CPU")
        else:
            logger.info(f"Loaded RoBERTa for classification with {num_labels} labels")
    elif task_type == "multiple_choice":
        num_choices = kwargs.pop("num_choices", 2)
        model = RobertaForMultipleChoiceZO(model_name, num_choices=num_choices)
        logger.info(f"Loaded RoBERTa for multiple choice with {num_choices} choices")
    else:
        raise ValueError(f"Unknown task type: {task_type}. Use 'classification', 'mlm', or 'multiple_choice'")

    return model

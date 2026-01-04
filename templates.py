"""
Template system for prompt-based learning.

This module provides:
1. Original Template classes for causal LM models (OPT, LLaMA, Mistral)
2. MeZO-style tokenize_multipart_input function for RoBERTa MLM tasks
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    task_name=None,
    prompt=False,
    template=None,
    label_word_list=None,
    first_sent_limit=None,
    other_sent_limit=None,
    gpt3=False,
    truncate_head=False,
    support_labels=None,
):
    """
    MeZO-style tokenization function for prompt-based learning.

    This function processes multi-part inputs with template-based prompts.
    Supports complex template variables like:
    - *cls*, *mask*, *sep*: Special tokens
    - *sent_i*: Input sentence i
    - *sentl_i*: Input sentence i with lowercase first letter
    - *+sent_i*: Input sentence i with space prefix
    - *label_i*: Label word i from label_word_list
    - *labelx_i*: Label word for support example i (for in-context learning)

    Args:
        input_text_list: List of input text strings
        max_length: Maximum sequence length
        tokenizer: HuggingFace tokenizer
        task_name: Task name for special handling
        prompt: Whether to use prompt-based format
        template: Template string with variables
        label_word_list: List of label word token IDs
        first_sent_limit: Max length for first sentence
        other_sent_limit: Max length for other sentences
        gpt3: Whether using GPT-3 style in-context learning
        truncate_head: Truncate from head instead of tail
        support_labels: Labels for support examples (ICL)

    Returns:
        Dict with keys: input_ids, attention_mask, token_type_ids (BERT only), mask_pos (if prompt=True)
    """
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = []  # Only for BERT
    mask_pos = None  # Position of the mask token

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = {
            'bos': tokenizer.bos_token_id, 'cls': tokenizer.cls_token_id, 'eos': tokenizer.eos_token_id,
            'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id,
        }
        template_list = template.split('*')  # Get variable list in the template
        segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.

        for part_id, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                # Check if tokenizer has cls/bos token before using it
                tokenizer_class = type(tokenizer).__name__
                has_model_type = hasattr(tokenizer, 'model_type')
                is_gpt2 = has_model_type and tokenizer.model_type == "gpt2"

                if (part == 'cls' or part == 'bos') and ('T5' in tokenizer_class or is_gpt2):
                    # T5 or GPT-2 do not have cls token
                    continue
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id])
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:8] == '+sentu-_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text[:-1])
            else:
                # Just natural language prompt
                part = part.replace('_', ' ')
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer.convert_tokens_to_ids(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    else:
        # Standard tokenization (non-prompt mode)
        if tokenizer.cls_token_id is not None:
            input_ids = [tokenizer.cls_token_id]
            attention_mask = [1]
            token_type_ids = [0]
        else:
            input_ids = []
            attention_mask = []
            token_type_ids = []

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__:  # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]

    # Padding is now handled by data collator (dynamic padding)
    # We don't pad to max_length here

    # Truncate
    if len(input_ids) > max_length:
        if first_sent_limit is not None:
            # If using sentence limit, the total length still exceeds the maximum limit, report a warning
            logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt and tokenizer.mask_token_id is not None:
        # Make sure that the masked position is inside the max_length
        assert tokenizer.mask_token_id in input_ids, \
            "Mask token not found for input: {} {}".format(input_text_list, input_ids)
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        assert mask_pos[0] < max_length
    elif prompt and tokenizer.mask_token_id is None:
        # autoregressive model
        mask_pos = [len(input_ids) - 1]

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__ or 'Roberta' in type(tokenizer).__name__:
        # Only provide token type ids for BERT/RoBERTa
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos

    return result


class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError

    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        return candidate

    def encode_sfc(self, sample):
        """
        Same as encode, but for SFC (calibration) -- this usually means the input is not included
        """
        return "<mask>"

    def verbalize_sfc(self, sample, candidate):
        """
        Same as verbalize, but for SFC (calibration) -- this usually means the input is not included
        """
        return candidate


class SST2Template(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return raw sentence only."""
        return sample.data["sentence"].strip()

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"

class SST2TemplateEmpty(Template):
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} "

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return raw sentence only."""
        return sample.data["sentence"].strip()

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f" "

    def verbalize_sfc(self, sample, candidate):
        return f" {self.verbalizer[candidate]}"


class SST2TemplateMLM(Template):
    """
    Simple MLM-style template for RoBERTa models (matching reference implementation).
    Format: "<sentence> It was <mask> ."
    The model predicts "bad" or "great" at the <mask> position.
    IMPORTANT: Both words are SINGLE tokens in RoBERTa tokenizer.
    """
    verbalizer = {0: "bad", 1: "great"}  # Both are single tokens: bad=10999, great=12338
    mask_token = "<mask>"  # Will be replaced with actual tokenizer mask token

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.mask_token} ."

    def verbalize(self, sample, candidate):
        """For MLM, we still use mask token during training."""
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.mask_token} ."

    def verbalize_for_label(self, sample, candidate):
        """Get the text with actual label for reference."""
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]} ."

    def encode_sfc(self, sample):
        return f"It was {self.mask_token} ."

    def verbalize_sfc(self, sample, candidate):
        return f"It was {self.mask_token} ."


class CopaTemplate(Template):
    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class CopaTemplateEmpty(Template):
    capitalization: str = "correct"
    effect_conj: str = " "
    cause_conj: str = " "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction

    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)

    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip()

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt


class BoolQTemplate(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return passage and question."""
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV2(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV3(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n{candidate}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on BoolQ task.
    Format: "<passage> <question> <mask> ."
    The model predicts "Yes" or "No" at the <mask> position.
    Label mapping: "Yes"=True, "No"=False
    Note: BoolQ uses string candidates ["Yes", "No"], not integer labels.
    """
    verbalizer = {"Yes": "Yes", "No": "No"}
    mask_token = "<mask>"

    def encode(self, sample):
        passage = sample.data["passage"].strip()
        question = sample.data["question"].strip()
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {self.mask_token} ."

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"].strip()
        question = sample.data["question"].strip()
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {self.mask_token} ."

    def verbalize_for_label(self, sample, candidate):
        passage = sample.data["passage"].strip()
        question = sample.data["question"].strip()
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {self.verbalizer[candidate]} ."

    def encode_sfc(self, sample):
        return f"{self.mask_token} ."

    def verbalize_sfc(self, sample, candidate):
        return f"{self.mask_token} ."


class MultiRCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n"

    def verbalize(self, sample, candidate):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class CBTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return premise and hypothesis as sentence pair."""
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class CBTemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on CB task.
    Format: "<premise> ? <mask> , <hypothesis>"
    The model predicts "Yes", "No", or "Maybe" at the <mask> position.
    Label mapping: 0=entailment(Yes), 1=contradiction(No), 2=neutral(Maybe)
    """
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}
    mask_token = "<mask>"

    def encode(self, sample):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize_for_label(self, sample, candidate):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.verbalizer[candidate]} , {hypothesis}"

    def encode_sfc(self, sample):
        return f"? {self.mask_token} ,"

    def verbalize_sfc(self, sample, candidate):
        return f"? {self.mask_token} ,"


class WICTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n"

    def verbalize(self, sample, candidate):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WICTemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on WIC task.
    Format: "<sent1> <sent2> Does the word "<word>" have the same meaning? <mask> ."
    The model predicts "No" or "Yes" at the <mask> position.
    Label mapping: 0=No (different meaning), 1=Yes (same meaning)
    """
    verbalizer = {0: "No", 1: "Yes"}
    mask_token = "<mask>"

    def encode(self, sample):
        sent1 = sample.data["sentence1"].strip()
        sent2 = sample.data["sentence2"].strip()
        word = sample.data["word"]
        return f"{sent1} {sent2} Does the word \"{word}\" have the same meaning? {self.mask_token} ."

    def verbalize(self, sample, candidate):
        sent1 = sample.data["sentence1"].strip()
        sent2 = sample.data["sentence2"].strip()
        word = sample.data["word"]
        return f"{sent1} {sent2} Does the word \"{word}\" have the same meaning? {self.mask_token} ."

    def verbalize_for_label(self, sample, candidate):
        sent1 = sample.data["sentence1"].strip()
        sent2 = sample.data["sentence2"].strip()
        word = sample.data["word"]
        return f"{sent1} {sent2} Does the word \"{word}\" have the same meaning? {self.verbalizer[candidate]} ."

    def encode_sfc(self, sample):
        return f"Does the word have the same meaning? {self.mask_token} ."

    def verbalize_sfc(self, sample, candidate):
        return f"Does the word have the same meaning? {self.mask_token} ."


class WSCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ReCoRDTemplate(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer:"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"Answer:"

    def verbalize_sfc(self, sample, candidate):
        return f"Answer: {candidate}"


class ReCoRDTemplateGPT3(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        return f"{passage}\n-"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"{passage}\n- {query}"

        # passage = sample.data['passage']
        # query = sample.data['query']
        # return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"-"

    def verbalize_sfc(self, sample, candidate):
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"- {query}"


class RTETemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return premise and hypothesis as sentence pair."""
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise} {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"

class RTETemplateEmpty(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\n\"{hypothesis}\"\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\n\"{hypothesis}\"\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class RTETemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on RTE task.
    Format: "<premise> ? <mask> , <hypothesis>"
    The model predicts "Yes" or "No" at the <mask> position.
    Label mapping: 0=entailment(Yes), 1=not_entailment(No)
    """
    verbalizer = {0: "Yes", 1: "No"}
    mask_token = "<mask>"

    def encode(self, sample):
        premise = sample.data['premise'].strip()
        hypothesis = sample.data['hypothesis'].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise'].strip()
        hypothesis = sample.data['hypothesis'].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize_for_label(self, sample, candidate):
        premise = sample.data['premise'].strip()
        hypothesis = sample.data['hypothesis'].strip()
        return f"{premise} ? {self.verbalizer[candidate]} , {hypothesis}"

    def encode_sfc(self, sample):
        return f"? {self.mask_token} ,"

    def verbalize_sfc(self, sample, candidate):
        return f"? {self.mask_token} ,"


class SQuADv2Template(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class DROPTemplate(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class WinoGrandeTemplate(Template):
    @staticmethod
    def get_prompt(sample):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = sample.data["sentence"]
        context, target = sentence.split("_")
        return context

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt

    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + candidate

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return candidate


# ============================================================================
# SNLI Templates (Stanford Natural Language Inference)
# ============================================================================

class SNLITemplate(Template):
    """
    Template for SNLI 3-class NLI task.
    Labels: 0=entailment, 1=neutral, 2=contradiction
    """
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\n"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return premise and hypothesis."""
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return self.verbalizer[candidate]


class SNLITemplateEmpty(Template):
    """Empty template for SNLI."""
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis} "

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return self.verbalizer[candidate]


class SNLITemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on SNLI task.
    Format: "<premise> ? <mask> , <hypothesis>"
    The model predicts "Yes", "Maybe", or "No" at the <mask> position.
    Label mapping: 0=entailment(Yes), 1=neutral(Maybe), 2=contradiction(No)
    """
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}
    mask_token = "<mask>"

    def encode(self, sample):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize_for_label(self, sample, candidate):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.verbalizer[candidate]} , {hypothesis}"

    def encode_sfc(self, sample):
        return f"? {self.mask_token} ,"

    def verbalize_sfc(self, sample, candidate):
        return f"? {self.mask_token} ,"


# ============================================================================
# MNLI Templates (Multi-Genre Natural Language Inference)
# ============================================================================

class MNLITemplate(Template):
    """
    Template for MNLI 3-class NLI task.
    Labels: 0=entailment, 1=neutral, 2=contradiction
    """
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\n"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification - return premise and hypothesis."""
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return self.verbalizer[candidate]


class MNLITemplateEmpty(Template):
    """Empty template for MNLI."""
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis} "

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"{premise} {hypothesis} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return self.verbalizer[candidate]


class MNLITemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on MNLI task.
    Format: "<premise> ? <mask> , <hypothesis>"
    The model predicts "Yes", "Maybe", or "No" at the <mask> position.
    Label mapping: 0=entailment(Yes), 1=neutral(Maybe), 2=contradiction(No)
    """
    verbalizer = {0: "Yes", 1: "Maybe", 2: "No"}
    mask_token = "<mask>"

    def encode(self, sample):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.mask_token} , {hypothesis}"

    def verbalize_for_label(self, sample, candidate):
        premise = sample.data["premise"].strip()
        hypothesis = sample.data["hypothesis"].strip()
        return f"{premise} ? {self.verbalizer[candidate]} , {hypothesis}"

    def encode_sfc(self, sample):
        return f"? {self.mask_token} ,"

    def verbalize_sfc(self, sample, candidate):
        return f"? {self.mask_token} ,"


# ============================================================================
# TREC Templates (Question Classification)
# ============================================================================

class TRECTemplate(Template):
    """
    Template for TREC 6-class question classification.
    Labels:
        0: ABBR (Abbreviation)
        1: ENTY (Entity)
        2: DESC (Description)
        3: HUM (Human)
        4: LOC (Location)
        5: NUM (Numeric)
    """
    verbalizer = {0: "Abbreviation", 1: "Entity", 2: "Description", 3: "Human", 4: "Location", 5: "Number"}

    def encode(self, sample):
        question = sample.data["question"]
        return f"Question: {question}\nType:"

    def encode_for_cls(self, sample):
        """For RoBERTa CLS classification."""
        question = sample.data["question"]
        return f"{question}"

    def verbalize(self, sample, candidate):
        question = sample.data["question"]
        return f"Question: {question}\nType: {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return "Type:"

    def verbalize_sfc(self, sample, candidate):
        return f"Type: {self.verbalizer[candidate]}"


class TRECTemplateEmpty(Template):
    """Empty template for TREC."""
    verbalizer = {0: "Abbreviation", 1: "Entity", 2: "Description", 3: "Human", 4: "Location", 5: "Number"}

    def encode(self, sample):
        question = sample.data["question"]
        return f"{question} "

    def verbalize(self, sample, candidate):
        question = sample.data["question"]
        return f"{question} {self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return ""

    def verbalize_sfc(self, sample, candidate):
        return self.verbalizer[candidate]


class TRECTemplateMLM(Template):
    """
    MLM-style template for RoBERTa models on TREC task.
    Format: "<question> This is a <mask> question."

    Using single-token verbalizer words:
        0: ABBR -> "short" (asking about abbreviations/expressions)
        1: ENTY -> "thing" (asking about entities)
        2: DESC -> "what" (asking for description/definition)
        3: HUM -> "who" (asking about humans)
        4: LOC -> "where" (asking about locations)
        5: NUM -> "how" (asking about numbers/amounts)
    """
    verbalizer = {0: "short", 1: "thing", 2: "what", 3: "who", 4: "where", 5: "how"}
    mask_token = "<mask>"

    def encode(self, sample):
        question = sample.data["question"].strip()
        return f"{question} This is a {self.mask_token} question."

    def verbalize(self, sample, candidate):
        question = sample.data["question"].strip()
        return f"{question} This is a {self.mask_token} question."

    def verbalize_for_label(self, sample, candidate):
        question = sample.data["question"].strip()
        return f"{question} This is a {self.verbalizer[candidate]} question."

    def encode_sfc(self, sample):
        return f"This is a {self.mask_token} question."

    def verbalize_sfc(self, sample, candidate):
        return f"This is a {self.mask_token} question."

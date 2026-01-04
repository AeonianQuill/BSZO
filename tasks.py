import logging
import sys
import json
from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np
from datasets import load_dataset

from templates import *
from utils import temp_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    """
    Original Sample class for causal LM models (OPT, LLaMA, Mistral).
    """
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None
    mask_pos: int = None  # Position of <mask> token for MLM-style tasks
    target_token_id: int = None  # Token ID of target word for MLM


@dataclass(frozen=True)
class OurInputFeatures:
    """
    MeZO-style input features for RoBERTa MLM tasks.

    This dataclass is used for prompt-based learning with RoBERTa models.
    It supports:
    - Mask position tracking for MLM
    - Label word lists for classification
    - SFC (Surface Form Calibration) inputs

    Frozen=True makes it hashable and immutable (required for some HF operations).
    """
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None  # Position of the mask token (List for compatibility)
    label_word_list: Optional[List[int]] = None  # Label word mapping (dynamic)

    # For ICL SFC (In-Context Learning Surface Form Calibration)
    sfc_input_ids: Optional[List[int]] = None
    sfc_attention_mask: Optional[List[int]] = None
    sfc_mask_pos: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        # Convert to dict first, handling None values
        data_dict = {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
            'label': self.label,
            'mask_pos': self.mask_pos,
            'label_word_list': self.label_word_list,
            'sfc_input_ids': self.sfc_input_ids,
            'sfc_attention_mask': self.sfc_attention_mask,
            'sfc_mask_pos': self.sfc_mask_pos,
        }
        return json.dumps(data_dict) + "\n"


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False  # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.samples = None
        self.subtask = subtask

    def get_task_name(self):
        return self.subtask

    def load_dataset(self, path, **kwargs):
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        return

    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            assert num_dev is None  # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:  # This is always False for now
                raise NotImplementedError
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed,
                                                            num=num_train + num_dev))  # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num + 1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: SST2Template, 1: SST2TemplateEmpty, 2: SST2TemplateMLM}[template_version]()


class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )

        return sample

    def get_template(self, template_version=0):
        return {0: CopaTemplate, 1: CopaTemplateEmpty}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )

        return sample

    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3, 3: BoolQTemplateMLM}[template_version]()


class MultiRCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: CBTemplate, 2: CBTemplateMLM}[template_version]()


class WICDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: WICTemplate, 2: WICTemplateMLM}[template_version]()


class WSCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: RTETemplate, 1: RTETemplateEmpty, 2: RTETemplateMLM}[template_version]()


class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class WinoGrandeDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_set = load_dataset('winogrande', 'winogrande_m', split='train')
        valid_set = load_dataset('winogrande', 'winogrande_m', split='validation')

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = example["sentence"]
        context, target = sentence.split("_")
        sample = Sample(
            data=example,
            candidates=[example['option1'] + target, example['option2'] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )
        return sample

    def get_template(self, template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        else:
            raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")


# ============================================================================
# MeZO-style MLM Dataset Classes for RoBERTa
# ============================================================================

class MLMDataset(Dataset):
    """
    Base class for MLM-style prompt-based learning (RoBERTa).

    This class extends the Dataset base class to support MeZO-style
    tokenization with template-based prompts for RoBERTa models.

    Key differences from standard Dataset:
    - Returns OurInputFeatures instead of Sample
    - Uses tokenize_multipart_input for complex template processing
    - Supports label word lists and mask position tracking
    """

    def __init__(self, subtask=None, **kwargs):
        super().__init__(subtask, **kwargs)
        self.label_to_word = None  # Will be set by subclass (e.g., {0: "bad", 1: "great"})
        self.num_labels = None  # Will be set by subclass

    def get_label_word_list(self, tokenizer):
        """
        Get label word list as token IDs.

        For RoBERTa, ensures that each label word is a SINGLE token.
        Applies space prefix if needed to ensure correct tokenization.

        Returns:
            List[int]: Token IDs for each label
        """
        if self.label_to_word is None:
            return None

        label_word_list = []
        for key in sorted(self.label_to_word.keys()):
            word = self.label_to_word[key]

            # Check if word starts with special characters (no space prefix needed)
            if word[0] not in ['<', '[', '.', ',']:
                # For RoBERTa, ensure "space + word" is a single token
                tokenized = tokenizer.tokenize(' ' + word)
                assert len(tokenized) == 1, \
                    f"Label word '{word}' (with space) is not a single token: {tokenized}"
                token_id = tokenizer.convert_tokens_to_ids(tokenized[0])
            else:
                # Special tokens or punctuation
                token_id = tokenizer.convert_tokens_to_ids(word)

            label_word_list.append(token_id)

        logger.info(f"Label word list for {self.__class__.__name__}: "
                    f"{[(k, self.label_to_word[k], label_word_list[k]) for k in sorted(self.label_to_word.keys())]}")

        return label_word_list

    def build_mlm_features(self, example, tokenizer, max_length, template_str, label_word_list=None):
        """
        Build OurInputFeatures for MLM-style tasks.

        Args:
            example: Raw dataset example
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
            template_str: Template string (e.g., "*cls**sent_0*It was*mask*.")
            label_word_list: List of label word token IDs

        Returns:
            OurInputFeatures
        """
        # Extract input text based on task
        input_text_list = self.extract_input_text(example)

        # Tokenize using MeZO-style function
        encoded = tokenize_multipart_input(
            input_text_list=input_text_list,
            max_length=max_length,
            tokenizer=tokenizer,
            task_name=self.get_task_name(),
            prompt=True,  # Always True for MLM
            template=template_str,
            label_word_list=label_word_list,
            first_sent_limit=None,
            other_sent_limit=None,
            truncate_head=False,
        )

        # Get label
        label = self.extract_label(example)

        # Create OurInputFeatures
        features = OurInputFeatures(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            token_type_ids=encoded.get('token_type_ids', None),
            label=label,
            mask_pos=encoded.get('mask_pos', None),
            label_word_list=label_word_list,
        )

        return features

    def extract_input_text(self, example):
        """
        Extract input text list from example.
        Should be overridden by subclass.

        Returns:
            List[str]: List of input sentences
        """
        raise NotImplementedError

    def extract_label(self, example):
        """
        Extract label from example.
        Should be overridden by subclass.

        Returns:
            int or float: Label
        """
        raise NotImplementedError


class SNLIDataset(Dataset):
    """
    SNLI (Stanford Natural Language Inference) dataset.

    3-class NLI task: entailment, neutral, contradiction
    Labels: 0=entailment, 1=neutral, 2=contradiction
    """
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('snli')
        # Filter out examples with label -1 (no gold label)
        train_d = d["train"].filter(lambda x: x["label"] != -1)
        validation_d = d["validation"].filter(lambda x: x["label"] != -1)

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example, idx=None):
        label = int(example["label"])
        return Sample(
            id=idx,
            data={
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
            },
            correct_candidate=label,
            candidates=[0, 1, 2]  # entailment, neutral, contradiction
        )

    def get_template(self, template_version=0):
        return {0: SNLITemplate, 1: SNLITemplateEmpty, 2: SNLITemplateMLM}[template_version]()


class MNLIDataset(Dataset):
    """
    MNLI (Multi-Genre Natural Language Inference) dataset.

    3-class NLI task: entailment, neutral, contradiction
    Labels: 0=entailment, 1=neutral, 2=contradiction

    Supports subtasks: 'matched' or 'mismatched' for validation
    """
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask  # 'matched' or 'mismatched'
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'mnli')
        train_d = d["train"]

        # Use matched or mismatched validation based on subtask
        if self.subtask == "mismatched":
            validation_d = d["validation_mismatched"]
        else:
            validation_d = d["validation_matched"]  # default to matched

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example, idx=None):
        label = int(example["label"])
        return Sample(
            id=idx,
            data={
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
            },
            correct_candidate=label,
            candidates=[0, 1, 2]  # entailment, neutral, contradiction
        )

    def get_template(self, template_version=0):
        return {0: MNLITemplate, 1: MNLITemplateEmpty, 2: MNLITemplateMLM}[template_version]()


class SST2MLMDataset(MLMDataset):
    """
    SST2 dataset for RoBERTa MLM-style training.

    Uses MeZO-style template: "*cls**sent_0*It was*mask*."
    Label words: {0: "bad", 1: "great"} (both are single tokens in RoBERTa)
    """
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs):
        super().__init__(subtask, **kwargs)
        # Label words must match SST2TemplateMLM.verbalizer
        # Using "bad" and "great" - both are single tokens in RoBERTa
        self.label_to_word = {0: "bad", 1: "great"}
        self.num_labels = 2
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]

        # Keep original Sample-based format for compatibility
        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """Build original Sample for backward compatibility."""
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def extract_input_text(self, example):
        """Extract sentence from SST2 example."""
        return [example["sentence"].strip()]

    def extract_label(self, example):
        """Extract label from SST2 example."""
        return int(example["label"])

    def get_template(self, template_version=0):
        """
        Return template class for SST2.
        Note: For MLM mode, we recommend using template_version=2 (SST2TemplateMLM)
        """
        return {0: SST2Template, 1: SST2TemplateEmpty, 2: SST2TemplateMLM}[template_version]()


class TRECDataset(Dataset):
    """
    TREC Question Classification dataset.

    6-class question type classification:
    - 0: ABBR (Abbreviation)
    - 1: ENTY (Entity)
    - 2: DESC (Description/Definition)
    - 3: HUM (Human)
    - 4: LOC (Location)
    - 5: NUM (Numeric)
    """
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('trec')
        train_d = d["train"]
        validation_d = d["test"]  # TREC uses test as validation

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example, idx=None):
        # TREC uses 'coarse_label' for 6-class classification
        label = int(example["coarse_label"])
        return Sample(
            id=idx,
            data={
                "question": example["text"],
                "label": label,
            },
            correct_candidate=label,
            candidates=[0, 1, 2, 3, 4, 5]
        )

    def get_template(self, template_version=0):
        return {0: TRECTemplate, 1: TRECTemplateEmpty, 2: TRECTemplateMLM}[template_version]()

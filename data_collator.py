import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertTokenizerFast
from transformers import PreTrainedTokenizerBase
from transformers.file_utils  import PaddingStrategy

from transformers import DataCollatorForTokenClassification


InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""


class DataCollatorForProbTokenClassification(DataCollatorForTokenClassification):
    """_summary_

    :param DataCollatorForTokenClassification: _description_
    :type DataCollatorForTokenClassification: _type_
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        #label_name = "label" if "label" in features[0].keys() else "labels"
        label_name = "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        label_probs = [feature["label_probs"] for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            batch["label_probs"] = [label_prob + [[0]*len(label_prob[0])] * (sequence_length - len(label_prob)) for label_prob in label_probs]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            batch["label_probs"] = [[[0]*len(label_prob[0])] * (sequence_length - len(label_prob)) + label_prob for label_prob in label_probs]

        batch = {
            k: torch.tensor(v, dtype=torch.int64) if k!="label_probs" else torch.tensor(v, dtype=torch.float32) 
            for k, v in batch.items()
        }
        return batch


def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def tolist(x: Union[List[Any], torch.Tensor]):
    return x.tolist() if isinstance(x, torch.Tensor) else x

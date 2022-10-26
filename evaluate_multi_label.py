import importlib
from typing import List, Optional, Union

from seqeval.metrics import accuracy_score, classification_report

import datasets

import numpy as np
from torch import nn
import torch

_DESCRIPTION = """Evaluation scripts for multi-label seq eval. 
Inputs: Prediction tensor (predictions) and True label tensor (references). 
Optional: Threshold value tuple (pred_t, ref_t)
Both tensors should have a shape of [# of sentences in the mini batch * # of words * # of labels]
Outputs: A dictionary of entity level F1 score for each entity type."""
_CITATION = """KAZU. Wonjin Yoon et al. and
Code modified from https://github.com/chakki-works/seqeval"""
_KWARGS_DESCRIPTION= """TBA"""

def B_I_O(B_labels, I_labels): # Our rule! : If B and I also exists in the label, use I
    if B_labels == "O" and I_labels == "O":
        return "O"
    elif B_labels != "O" and I_labels != "O":
        return I_labels
    elif B_labels == "O" and I_labels != "O":
        return I_labels
    elif B_labels != "O" and I_labels == "O":
        return B_labels

def seqeval_prob_compute(
    predictions,
    references,
    suffix: bool = False,
    scheme: Optional[str] = None,
    mode: Optional[str] = None,
    sample_weight: Optional[List[int]] = None,
    zero_division: Union[str, int] = "warn",
    label_list: Optional[list] = None,
):
    """_summary_

    :param predictions: Prediction tensor (predictions).
    :type predictions: list or tensor. A shape of [# of sentences in the mini batch * # of words * # of labels]
    :param references: True label tensor (references). 
    :type references: list or tensor. A shape of [# of sentences in the mini batch * # of words * # of labels]
    :param suffix: _description_, defaults to False
    :type suffix: bool, optional
    :param scheme: _description_, defaults to None
    :type scheme: Optional[str], optional
    :param mode: _description_, defaults to None
    :type mode: Optional[str], optional
    :param sample_weight: _description_, defaults to None
    :type sample_weight: Optional[List[int]], optional
    :param zero_division: _description_, defaults to "warn"
    :type zero_division: Union[str, int], optional
    :param label_list: List of labels. 
    :type label_list: Optional[list], optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    # Eval script for multi-label NER. Code modified from "https://github.com/chakki-works/seqeval".
    if scheme is not None:
        try:
            scheme_module = importlib.import_module("seqeval.scheme")
            scheme = getattr(scheme_module, scheme)
        except AttributeError:
            raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")

    if label_list == None:
        label_list = list(range(len(predictions[0])))

    entity_types = set([lab.split("-")[-1] for lab in label_list])
    results_by_entity_type = {
        ent_type:{
            "predictions":{"B":[], "I":[]},
            "references":{"B":[], "I":[]}
        } 
        for ent_type in entity_types
    }
    

    total_report = dict() # btw, what is mode?
    for entity_idx, entity_type_w_label in enumerate(label_list):
        if entity_type_w_label == "O":
            continue
        elif "-" in entity_type_w_label:
            entity_type = entity_type_w_label.split("-")[-1]
        else:
            print(f"{entity_type_w_label}: Wrong! Should be O, B-, I-.") # for debug
            continue

        pred_for_type = [
            [("I" if logits[entity_idx]>0 else "O") for logits in prediction_sent] # I if logit>0 
            for prediction_sent in predictions
        ]
        ref_for_type = [
            [("I" if probs[entity_idx]==1 else "O") for probs in reference_sent]
            for reference_sent in references
        ]
        """
        #import pdb;pdb.set_trace()
        sigmoid = nn.Sigmoid()
        maybe_pred_for_type = [
            [("I" if sigmoid(torch.tensor(logits[entity_idx]))>0.5 else "O") for logits in prediction_sent] # I if logit>0 
            for prediction_sent in predictions
        ]"""
        if "B" == entity_type_w_label.split("-")[0]:
            results_by_entity_type[entity_type]["predictions"]["B"] = [
                [("B" if logits[entity_idx]>0 else "O") for logits in prediction_sent] # I if logit>0 
                for prediction_sent in predictions
            ]
            results_by_entity_type[entity_type]["references"]["B"] = [
                [("B" if probs[entity_idx]==1 else "O") for probs in reference_sent]
                for reference_sent in references
            ]
        elif "I" == entity_type_w_label.split("-")[0]:
            results_by_entity_type[entity_type]["predictions"]["I"] = [
                [("I" if logits[entity_idx]>0 else "O") for logits in prediction_sent] # I if logit>0 
                for prediction_sent in predictions
            ]
            results_by_entity_type[entity_type]["references"]["I"] = [
                [("I" if probs[entity_idx]==1 else "O") for probs in reference_sent]
                for reference_sent in references
            ]
    
    print("### Eval results: ")
    for entity_type, values in results_by_entity_type.items():
        if entity_type == "O":
            continue
        #import pdb;pdb.set_trace()
        if len(values["predictions"]["B"]) == 0:
            values["predictions"]["B"] = values["predictions"]["I"]
        if len(values["references"]["B"]) == 0:
            values["references"]["B"] = values["references"]["I"]

        ref_for_type = [
            [B_I_O(ref_B, ref_I) for ref_B, ref_I in zip(reference_sent_B, reference_sent_I)]
            for reference_sent_B, reference_sent_I in zip(values["references"]["B"], values["references"]["I"])
        ]
        pred_for_type = [
            [B_I_O(prde_B, pred_I) for prde_B, pred_I in zip(prediction_sent_B, prediction_sent_I)]
            for prediction_sent_B, prediction_sent_I in zip(values["predictions"]["B"], values["predictions"]["I"])
        ]
        report = classification_report(
            y_true=ref_for_type,
            y_pred=pred_for_type,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        #overall_score = report.pop("micro avg")

        total_report[entity_type] = {
            f"precision": report['_']["precision"],
            f"recall": report['_']["recall"],
            f"f1": report['_']["f1-score"],
            f"number": report['_']["support"],
            f"accuracy": accuracy_score(y_true=ref_for_type, y_pred=pred_for_type)
        }
        easy_read_report = ", ".join([f"{k}: {v:.4f}" for k,v in total_report[entity_type].items()])
        print(f"# {entity_type:12s}: {easy_read_report}") # for debug
        
    scores = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "number": score["number"],
            "accuracy": score["accuracy"],
        }
        for type_name, score in total_report.items()
    }

    # overall_score
    score_sum = 0
    scores["overall_precision"] = 0
    scores["overall_recall"] = 0
    scores["overall_f1"] = 0
    scores["overall_accuracy"] = 0

    for type_name, score in total_report.items():
        score_sum += score["number"]
        scores["overall_precision"] += score["precision"]
        scores["overall_recall"] += score["recall"]
        scores["overall_f1"] += score["f1"]
        scores["overall_accuracy"] += score["accuracy"]

    scores["overall_precision"] = scores["overall_precision"]/score_sum
    scores["overall_recall"] = scores["overall_recall"]/score_sum
    scores["overall_f1"] = scores["overall_f1"]/score_sum
    scores["overall_accuracy"] = scores["overall_accuracy"]/score_sum

    return scores


def use_original_seqeval_prob_compute(
    predictions,
    references,
    suffix: bool = False,
    scheme: Optional[str] = None,
    mode: Optional[str] = None,
    sample_weight: Optional[List[int]] = None,
    zero_division: Union[str, int] = "warn",
    label_list: Optional[list] = None,
):
    # Not a multi-label eval code.
    predictions = [
            [label_list[np.argmax(p)] for p in prediction]
            for prediction in predictions
    ]
    references = [
            [label_list[np.argmax(p)] for p in reference]
            for reference in references
    ]
    # Eval script for multi-label NER. Code modified from "https://github.com/chakki-works/seqeval".
    if scheme is not None:
        try:
            scheme_module = importlib.import_module("seqeval.scheme")
            scheme = getattr(scheme_module, scheme)
        except AttributeError:
            raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")

    report = classification_report(
        y_true=references,
        y_pred=predictions,
        suffix=suffix,
        output_dict=True,
        scheme=scheme,
        mode=mode,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    scores = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "number": score["support"],
        }
        for type_name, score in report.items()
    }
    scores["overall_precision"] = overall_score["precision"]
    scores["overall_recall"] = overall_score["recall"]
    scores["overall_f1"] = overall_score["f1-score"]
    scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)

    return scores




class MultiLabelSeqEval(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    #"predictions": {"id": datasets.Value("string"), "prediction_text": datasets.Value("string")},
                    "predictions": datasets.Sequence(datasets.Sequence(datasets.Value(dtype="float32", id="probs"))),
                    "references": datasets.Sequence(datasets.Sequence(datasets.Value(dtype="float32", id="probs"))),
                }
            ),
            codebase_urls=["TBA"],
            reference_urls=["TBA"],
        )

    def _compute(
        self,
        predictions,
        references,
        suffix: bool = False,
        scheme: Optional[str] = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = "warn",
        label_list: Optional[list] = None,
    ):
        seqeval_prob_compute(
            predictions=predictions,
            references=references,
            suffix=suffix,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
            label_list=label_list,
        )
        scores = use_original_seqeval_prob_compute(
            predictions=predictions,
            references=references,
            suffix=suffix,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
            label_list=label_list,
        )

        return scores

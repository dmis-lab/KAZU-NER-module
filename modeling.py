#!/usr/bin/env python
# coding=utf-8

import logging
from typing import Optional

import torch
from torch import nn

import numpy as np

import transformers
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from transformers.utils import check_min_version

check_min_version("4.9.0") # transformers version check

class BertForMultiLableTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.ignore_index = -100
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_probs=None, # for soft-label (prob-based) training!
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length, self.num_labels)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, 1]``.
            Each token has a vector of self.num_labels float values.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if label_probs is not None:
            loss_fct = nn.BCELoss()
            sigmoid = nn.Sigmoid()
            
            loss = loss_fct(sigmoid(logits.view(-1, self.num_labels)), 
                            label_probs.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

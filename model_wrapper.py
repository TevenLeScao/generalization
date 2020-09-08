import os

import numpy as np
import torch
from torch import nn as nn
from transformers import BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, PreTrainedModel, \
    Trainer

from utils import combine_encodings


class ModelWrapper(nn.Module):
    def __init__(self, model_type, model_name, device='cpu', token_limit=None):
        super().__init__()
        MODEL_CLASSES = {
            'bert': (BertForMaskedLM, BertTokenizerFast),
            'roberta': (RobertaForMaskedLM, RobertaTokenizerFast),
        }
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=('uncased' in model_name))
        self.lm = model_class.from_pretrained(model_name, return_dict=True)
        self.transformer = getattr(self.lm, model_type)
        self.model_type = model_type
        self.device = device

        self.dim = self.transformer.pooler.dense.in_features
        self.max_len = self.transformer.embeddings.position_embeddings.num_embeddings

        self.unmasked_label_id = -100
        self.token_limit = token_limit

    def _prepare_inputs(self, inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        return inputs

    def bert_features(self, hypotheses, premises):
        """
        Input: list of sentences or sentence pairs
        Output: tensor (len(sentences), bert_dim) with sentence representations, or
            tensor (num_words_packed, bert_dim) with word representations
        """
        encoding = self.tokenizer(text=hypotheses, text_pair=premises, padding=True, pad_to_multiple_of=8,
                                  return_tensors="pt")
        encoding = self._prepare_inputs(encoding)
        features = self.transformer(encoding.data["input_ids"], attention_mask=encoding.data["attention_mask"])[
            "pooler_output"]
        return features

    def encode_outputs_with_conditioning(self, previous_premises, previous_hypotheses):
        previous_sentences = [[premise + " " + hypothesis for premise, hypothesis in zip(premises, hypotheses)] for
                              premises, hypotheses in zip(previous_premises, previous_hypotheses)]
        if self.token_limit is not None:
            for sentences in previous_sentences:
                final_encoding = None
                total_tokens = 0
                for sentence in sentences:
                    encoding = self.tokenizer([sentence], return_tensors="pt")
                    n_tokens = encoding.data['input_ids'].shape[1]
                    if total_tokens + n_tokens < self.token_limit:
                        final_encoding = combine_encodings(final_encoding, encoding)
                    else:
                        break



    def predict_mlm(self, premises, hypotheses, previous_premises=None, previous_hypotheses=None):
        """
        Input: tuple of lists of sentences, which are lists of words.
        Output: tensor (num_masked, vocab_size) with prediction logits
        """
        sentences = [premise + " " + hypothesis for premise, hypothesis in zip(premises, hypotheses)]
        encoding = self.tokenizer(sentences, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        encoding = self._prepare_inputs(encoding)
        label_mask = encoding.data["input_ids"].clone()
        label_mask[label_mask != self.tokenizer.mask_token_id] = self.unmasked_label_id
        logits = self.lm(encoding.data["input_ids"], attention_mask=encoding.data["attention_mask"],
                         labels=label_mask)['logits']
        logits = logits.masked_select((label_mask != self.unmasked_label_id).unsqueeze(-1)).reshape(-1,
                                                                                                    logits.shape[-1])
        return logits

    def reset_weights(self, encoder_only=True):
        for name, module in self.named_modules():
            if hasattr(module, 'reset_parameters') and ('encoder' in name or not encoder_only):
                module.reset_parameters()
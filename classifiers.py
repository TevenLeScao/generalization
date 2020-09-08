import torch
from torch import nn as nn
from torch.nn import functional as F

from model_wrapper import ModelWrapper


class MLPClassifier(nn.Module):
    def __init__(self, model: ModelWrapper, num_classes, device='cpu'):
        super().__init__()
        self.model = model
        self.model_type = model.model_type
        self.dropout = nn.Dropout(p=0.1)
        self.span_tip = nn.Linear(self.model.dim, num_classes)
        self.device = device

    def forward(self, hypotheses, premises):
        embs = self.model.bert_features(hypotheses, premises)
        return self.span_tip(self.dropout(embs))

    def freeze_head(self, freeze=True):
        for param in self.span_tip.parameters():
            param.requires_grad = not freeze


class MLMClassifier(nn.Module):
    def __init__(self, model: ModelWrapper, classes, device='cpu',
                 token_limit=None):  # ex: [['he', 'him', 'himself'], ['she', 'her', 'herself']]
        super().__init__()
        self.model = model
        self.model_type = model.model_type
        self.classes = classes
        self.class_ids = []
        for lst in classes:
            ids = self.model.tokenizer.convert_tokens_to_ids(lst)
            assert self.model.tokenizer.unk_token_id not in ids
            self.class_ids.append(ids)

        self.device = device
        self.token_limit = token_limit

    def forward(self, premises, hypotheses, previous_premises=None, previous_hypotheses=None):
        # sentences are lists of words, label_masks are lists of 0/1 where 1 = masked
        assert len(premises) == len(hypotheses), \
            f"{len(premises)} premises =/= {len(hypotheses)} hypotheses"
        output_logits = self.model.predict_mlm(premises, hypotheses, previous_premises, previous_hypotheses)
        assert len(output_logits) == len(hypotheses), \
            f"Masked words should be one subword, but num_masks {len(output_logits)} =/= num_sent {len(hypotheses)}"
        output_logits = F.log_softmax(output_logits, dim=-1)
        class_logits = torch.cat(
            tuple(torch.logsumexp(output_logits[:, ids], dim=-1).unsqueeze(-1) for ids in self.class_ids), dim=-1)
        return class_logits

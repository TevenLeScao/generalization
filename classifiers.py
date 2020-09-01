import torch
from torch import nn as nn
from torch.nn import functional as F

from sentence_or_word_bert import ModelWrapper
from utils import use_cuda



class MLPClassifier(nn.Module):
    def __init__(self, model: ModelWrapper, num_classes):
        super().__init__()
        self.model = model
        self.model_type = model.model_type
        self.dropout = nn.Dropout(p=0.1)
        self.span_tip = nn.Linear(self.model.dim, num_classes)

        if use_cuda:
            self.cuda()

    def forward(self, sentences, word_level=False):
        embs, _, _ = self.model.annotate(sentences, word_level=word_level)
        return self.span_tip(self.dropout(embs))

    def freeze_head(self, freeze=True):
        for param in self.span_tip.parameters():
            param.requires_grad = not freeze


class MLMClassifier(nn.Module):
    def __init__(self, model: ModelWrapper, classes):  # ex: [['he', 'him', 'himself'], ['she', 'her', 'herself']]
        super().__init__()
        self.model = model
        self.model_type = model.model_type
        self.classes = classes
        self.class_ids = []
        for lst in classes:
            ids = [self.model.tokenizer.convert_tokens_to_ids(word) for word in lst]
            assert self.model.tokenizer.unk_token_id not in ids
            self.class_ids.append(ids)

        if use_cuda:
            self.cuda()

    def forward(self, exmps):
        # sentences are lists of words, label_masks are lists of 0/1 where 1 = masked
        sentences, label_masks = [e[0] for e in exmps], [e[1] for e in exmps]
        if isinstance(label_masks[0], tuple):
            assert all([sum(mask1 + mask2) == 1 for mask1, mask2 in
                        label_masks]), "There should be one masked word per sentence pair"
        else:
            assert all([sum(mask) == 1 for mask in label_masks]), "There should be one masked word per sentence"
        output_logits, _, _ = self.model.predict_mlm(sentences, label_masks)
        assert len(output_logits) == len(sentences), \
            "Masked words should be one subword, but num_masks {} =/= num_sent {}".format(len(output_logits),
                                                                                          len(sentences))
        output_logits = F.log_softmax(output_logits, dim=-1)
        class_logits = torch.cat(
            tuple(torch.logsumexp(output_logits[:, ids], dim=-1).unsqueeze(-1) for ids in self.class_ids), dim=-1)
        return class_logits
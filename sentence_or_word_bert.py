import numpy as np
import torch
from nltk import word_tokenize
from torch import nn as nn
from transformers import BertForMaskedLM, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer

from utils import use_cuda, from_numpy


class SentenceOrWordBert(nn.Module):
    def __init__(self, model_type, model_name):
        super().__init__()
        MODEL_CLASSES = {
            'bert': (BertForMaskedLM, BertTokenizer),
            'roberta': (RobertaForMaskedLM, RobertaTokenizer),
        }
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=('uncased' in model_name))
        self.bert_lm = model_class.from_pretrained(model_name)
        self.bert = self.bert_lm.bert
        self.model_type = model_type

        self.dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings

        self.unmasked_label_id = -100

        if use_cuda:
            self.cuda()

    def tokenize(self, sent, label_mask, include_clssep=False):
        """
        sent - string or list of words
        label_mask - list of either 0 or 1, 1 for masked (same length as sent)
        include_clssep - whether or not to include [CLS] and [SEP] in end_mask

        input_ids - sent converted to ids, with some ids masked
        end_mask - each word might be multiple subwords, so end_mask is 1 for
            the final subword of each word
        label_ids - -100 for unmasked tokens, the label for masked tokens
        """
        if isinstance(sent, str):
            sent = word_tokenize(sent)
        if label_mask is None:
            label_mask = [0 for _ in sent]

        input_ids = [self.tokenizer.cls_token_id]
        end_mask = [int(include_clssep)]
        label_ids = [self.unmasked_label_id]
        for word, is_masked in zip(sent, label_mask):
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            assert len(ids) > 0, "Unknown word {} in {}".format(word, sent)
            if is_masked:
                input_ids.extend([self.tokenizer.mask_token_id for _ in ids])
                label_ids.extend(ids)
            else:
                input_ids.extend(ids)
                label_ids.extend([self.unmasked_label_id for _ in ids])
            end_mask.extend([0 for _ in ids])
            end_mask[-1] = 1
        input_ids.append(self.tokenizer.sep_token_id)
        end_mask.append(int(include_clssep))
        label_ids.append(self.unmasked_label_id)
        return input_ids, end_mask, label_ids

    def tokenize_sentences(self, sentences, include_clssep=False, label_masks=None):
        """
        sentences - list of sentences, or tuples containing 2 sentences each
            each sentence is either a string a list of words
        """
        paired = isinstance(sentences[0], tuple) and len(sentences[0]) == 2
        if label_masks is None:
            if paired:
                label_masks = [(None, None) for _ in sentences]
            else:
                label_masks = [None for _ in sentences]

        all_input_ids = np.zeros((len(sentences), self.max_len), dtype=int) + self.tokenizer.pad_token_id
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype=int)
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype=int)
        all_label_ids = np.zeros((len(sentences), self.max_len), dtype=int) + self.unmasked_label_id

        max_sent = 0
        for s_num, (sent, label_mask) in enumerate(zip(sentences, label_masks)):
            if paired:
                input_ids1, end_mask1, label_ids1 = self.tokenize(sent[0], label_mask[0], include_clssep=include_clssep)
                input_ids2, end_mask2, label_ids2 = self.tokenize(sent[1], label_mask[1], include_clssep=include_clssep)

                input_ids = input_ids1 + input_ids2[1:]  # [cls] sent1 [sep] sent2 [sep]
                end_mask = end_mask1 + end_mask2[1:]
                label_ids = label_ids1 + label_ids2[1:]
            else:
                input_ids, end_mask, label_ids = self.tokenize(
                    sent, label_mask, include_clssep=include_clssep)

            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(input_ids)] = end_mask
            all_label_ids[s_num, :len(input_ids)] = label_ids
            max_sent = max(max_sent, len(input_ids))

        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :max_sent]))
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :max_sent]))
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask[:, :max_sent])).to(torch.uint8)
        all_label_ids = from_numpy(np.ascontiguousarray(all_label_ids[:, :max_sent]))
        return all_input_ids, all_input_mask, all_end_mask, all_label_ids

    def run_bert(self, all_input_ids, all_input_mask, subbatch_size=64):
        """
        all_input_ids, all_input_mask - tensors (batch, maxlen)
        features_all - tensor (batch, maxlen, dim)
        hidden_all - tuple of tensors (batch, maxlen, dim), one for embedding and one for each layer
        attentions_all - tuple of tensors (batch, heads, maxlen, maxlen), one for each layer
        """
        features_all = None
        attentions_all = None
        for i in range(0, len(all_input_ids), subbatch_size):
            input_ids = all_input_ids[i:i + subbatch_size]
            input_mask = all_input_mask[i:i + subbatch_size]

            # (batch, len, dim or vocab_size), tuple((batch, heads, len, len) x layers)
            features, _, attentions = self.bert(input_ids, attention_mask=input_mask, output_attentions=True)

            if features_all is None:
                features_all = features
                attentions_all = list(attentions)
            else:
                features_all = torch.cat((features_all, features), dim=0)
                for i, (attn_all, attn) in enumerate(zip(attentions_all, attentions)):
                    attentions_all[i] = torch.cat((attn_all, attn), dim=0)
        return features_all, attentions_all

    def annotate(self, sentences, include_clssep=False, word_level=False, subbatch_size=64):
        """
        Input: list of sentences or sentence pairs
            include_clssep - whether or not to include [CLS] and [SEP] in the end_mask
        Output: tensor (len(sentences), bert_dim) with sentence representations, or
            tensor (num_words_packed, bert_dim) with word representations
        """
        all_input_ids, all_input_mask, all_end_mask, _ = \
            self.tokenize_sentences(sentences, include_clssep=include_clssep)
        features, attn = self.run_bert(all_input_ids, all_input_mask, subbatch_size=subbatch_size)
        if word_level:
            features = features.masked_select(all_end_mask.unsqueeze(-1)).reshape(-1, features.shape[-1])
        else:
            features = features[:, 0]
        return features, all_input_ids, attn

    def run_bert_lm(self, all_input_ids, all_input_mask, all_label_ids, subbatch_size=64):
        """
        all_input_ids, all_input_mask, all_label_ids - tensors (batch, maxlen)
        loss - 0-dim tensor
        logits_all - tensor (num_masked, vocabsize)
        hidden_all - tuple of tensors (batch, maxlen, dim), one for embedding and one for each layer
        attentions_all - tuple of tensors (batch, heads, maxlen, maxlen), one for each layer
        """
        loss_sum, n = None, 0
        logits_all = None
        attentions_all = None
        for i in range(0, len(all_input_ids), subbatch_size):
            input_ids = all_input_ids[i:i + subbatch_size]
            input_mask = all_input_mask[i:i + subbatch_size]
            label_ids = all_label_ids[i:i + subbatch_size]

            loss, logits, attentions = self.bert_lm(input_ids, attention_mask=input_mask, labels=label_ids,
                                                    output_attentions=True)
            logits = logits.masked_select((label_ids != -100).unsqueeze(-1)).reshape(-1, logits.shape[-1])

            if loss_sum is None:
                loss_sum, n = loss, 1
                logits_all = logits
                attentions_all = list(attentions)
            else:
                loss_sum, n = loss_sum + loss, n + 1
                logits_all = torch.cat((logits_all, logits), dim=0)
                for i, (attn_all, attn) in enumerate(zip(attentions_all, attentions)):
                    attentions_all[i] = torch.cat((attn_all, attn), dim=0)
        return loss_sum / n, logits_all, attentions_all

    def predict_mlm(self, sentences, label_masks, output_hidden=False, subbatch_size=64):
        """
        Input: list of sentences, which are lists of words.
            list of masks, which are lists of either 0 or 1, 1 for masked
        Output: tensor (num_masked, vocab_size) with prediction logits
        """
        all_input_ids, all_input_mask, _, all_label_ids = self.tokenize_sentences(sentences, label_masks=label_masks)
        _, logits_all, attn = self.run_bert_lm(all_input_ids, all_input_mask, all_label_ids,
                                               subbatch_size=subbatch_size)
        return logits_all, all_input_ids, attn

    def reset_weights(self, encoder_only=True):
        for name, module in self.named_modules():
            if hasattr(module, 'reset_parameters') and ('encoder' in name or not encoder_only):
                module.reset_parameters()
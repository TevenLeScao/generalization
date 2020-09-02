from copy import deepcopy

import numpy as np
import torch
from nlp import load_dataset
from tqdm.notebook import tqdm
from transformers import XLNetLMHeadModel, AutoTokenizer

np.random.seed(314)
train_full = load_dataset('winogrande', 'winogrande_xl', split='train')
train_full = train_full.shuffle()
inds = np.arange(len(train_full))
split_loc = int(len(inds) * .8)
train = train_full.select(inds[:split_loc])
val = train_full.select(inds[split_loc:])

device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
model = XLNetLMHeadModel.from_pretrained("xlnet-large-cased", mem_len=2**14).to(device)

for p in model.parameters():
    p.requires_grad = False
    p.grad = None


def create_pred_encodings(encodings, pred_loc):
    """ Create encodings needed for XLNet """
    encodings = deepcopy(encodings)
    encodings['input_ids'][:, pred_loc] = tokenizer.mask_token_id
    encodings['attention_mask'][:, pred_loc] = 0.

    seqlen = encodings.input_ids.size(1)
    perm_mask = torch.zeros((1, seqlen, seqlen))
    perm_mask[:, :, pred_loc[0]] = 1.0  # todo: make sure this is right
    encodings['perm_mask'] = perm_mask

    target_mapping = torch.zeros((1, 1, seqlen))
    target_mapping[:, :, pred_loc[0]] = 1.
    encodings['target_mapping'] = target_mapping

    return encodings


def long_seq_pass(model, encodings):
    """ Get outputs when output is greater than 1024 """
    max_len = model.base_model.mem_len
    num_passes = encodings.input_ids.size(1) // max_len + 1
    mems = None
    for i in range(num_passes):
        ran = slice(i * max_len, (i + 1) * max_len)
        trunc_encodings = {
            'input_ids': encodings.input_ids[:, ran].to(device),
            'token_type_ids': encodings.token_type_ids[:, ran].to(device),
            'attention_mask': encodings.attention_mask[:, ran].to(device),
            'target_mapping': encodings.target_mapping[..., ran].to(device),
            'perm_mask': encodings.perm_mask[..., ran, ran].to(device)
        }
        if trunc_encodings['perm_mask'].bool().any():
            trunc_encodings['labels'] = encodings.labels.to(device)
        outputs = model(**trunc_encodings, mems=mems)
        mems = outputs[-1]
    return outputs


def get_ppl(model, cloze_sequence, target, context_examples=[]):
    """ Calculate the "perplexity" of the target subsequence given the sequence and context examples """
    # create context
    replaced_examples = []
    for ex in context_examples:
        label = int(ex['answer']) - 1
        options = [ex['option1'], ex['option2']]
        replaced_examples.append(ex['sentence'].replace('_', options[label]))
    context = ' '.join(replaced_examples)

    # create target encoding
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    repl_id = tokenizer.encode('_', add_special_tokens=False)[0]
    encodings = tokenizer(context, cloze_sequence, return_tensors='pt')
    target_start = torch.where(encodings.input_ids[0] == repl_id)[0]
    target_locs = list(range(target_start, target_start + len(target_ids)))

    outputs = None
    lls = []
    for trg_loc, trg in zip(target_locs, target_ids):
        trg_loc = list(range(trg_loc, target_locs[-1] + 1))
        pred_encodings = create_pred_encodings(encodings, trg_loc).to(device)
        pred_encodings['labels'] = torch.tensor(trg).view(-1)
        outputs = long_seq_pass(model, pred_encodings)
        lls.append(outputs[0])

    return (torch.mean(torch.stack(lls)),) + outputs[1:]


num_reps = 1  # number of times to replicate the experiment
accs = {n_train: [] for n_train in [0, 1, 50]}  # number of training examples to try
num_eval = 500

for rep in range(num_reps):
    val = val.shuffle(seed=np.random.randint(100000))  # workaround bc of seed bug in nlp
    for num_train in accs:
        print('Number of training points:', num_train)
        preds = []
        labels = []
        it = tqdm(val, miniters=5)
        for iexample, example in enumerate(it):
            context_examples = [train[int(i)] for i in np.random.choice(len(train), num_train, replace=False)]

            first_outputs = get_ppl(model, example['sentence'], example['option1'], context_examples)
            second_outputs = get_ppl(model, example['sentence'], example['option2'], context_examples)
            first_loss = first_outputs[0]
            second_loss = second_outputs[0]

            pred = torch.stack([first_loss, second_loss]).argmin()
            preds.append(pred.item())
            labels.append(int(example['answer']) - 1)

            acc = (np.array(preds) == np.array(labels)).mean()
            if iexample >= 0 and iexample % 5 == 0:
                it.set_description(f'acc: {acc * 100:0.4f}%')

            if (iexample + 1) % num_eval == 0:
                break

        print(num_train, acc)
        accs[num_train].append(acc)

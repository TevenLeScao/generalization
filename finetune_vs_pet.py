import json
import os

import nlp
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
from argparsing import parser

import torch
import torch.nn.functional as F

from classifiers import MLPClassifier, MLMClassifier
from utils import from_numpy
from model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)
nltk.download('punkt')


def evaluate(model, eval_data, subbatch_size=64, hans=False):
    model.eval()
    with torch.no_grad():
        preds = None
        for j in range(0, len(eval_data), subbatch_size):
            examples = eval_data[j:j + subbatch_size]
            logits = model([exmp[0] for exmp in examples])
            preds_batch = np.argmax(logits.cpu().numpy(), axis=1)
            if preds is None:
                preds = preds_batch
            else:
                preds = np.concatenate((preds, preds_batch), axis=0)
            if hans:
                preds = np.clip(preds, 0, 1)
        eval_acc = np.sum(np.array([exmp[1] for exmp in eval_data]) == preds) / len(eval_data)
    return eval_acc


def train(model, train_data, dev_data, hans_data, output_dir, lr_base=3e-5, lr_warmup_frac=0.1, epochs=5, batch_size=32,
          subbatch_size=8, eval_batch_size=64, check_every=2048, initial_check=False, verbose=True):
    print("lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}, len(train_data): {}".format(
        lr_base, lr_warmup_frac, epochs, batch_size, len(train_data)))

    params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    trainer = torch.optim.Adam([
        {'params': params, 'lr': 0., 'lr_base': lr_base, 'name': model.model_type}, ], lr=0.)

    def set_lr(lr_ratio):
        for param_group in trainer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    log = []
    processed = 0
    if initial_check:
        check_processed = check_every
    else:
        check_processed = 0
    train_acc_sum, train_acc_n = 0, 0
    best_dev_acc = 0
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(train_data)
        for i in tqdm(range(0, len(train_data), batch_size)):
            examples = train_data[i:i + batch_size]
            if len(examples) == batch_size:
                model.train()
                trainer.zero_grad()

                if check_processed >= check_every:
                    dev_acc = evaluate(model, dev_data, eval_batch_size)
                    hans_acc = evaluate(model, hans_data, eval_batch_size, hans=True)
                    train_acc = train_acc_sum / train_acc_n if train_acc_n > 0 else None
                    log.append({'dev_acc': dev_acc,
                                'hans_acc': hans_acc,
                                'train_acc': train_acc})
                    train_acc_sum, train_acc_n = 0, 0
                    check_processed -= check_every
                    if verbose:
                        print("Epoch: {}, Log: {}".format(epoch, log[-1]))
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        torch.save(model, os.path.join(output_dir, f"best_{model.model_type}"))

                for j in range(0, len(examples), subbatch_size):
                    examples_subbatch = examples[j:j + subbatch_size]

                    # compute loss, also log other metrics
                    logits = model([exmp[0] for exmp in examples_subbatch])
                    labels = np.array([exmp[1] for exmp in examples_subbatch])
                    loss = F.cross_entropy(logits, from_numpy(labels))
                    loss.backward()
                    del loss

                    batch_acc = np.sum(labels == np.argmax(logits.detach().cpu().numpy(), axis=1)) / len(labels)
                    train_acc_sum += batch_acc
                    train_acc_n += 1

                trainer.step()
                processed += len(examples)
                check_processed += len(examples)

                # warmup from 0 to lr_base for lr_warmup_frac
                lr_ratio = min(1, processed / (lr_warmup_frac * epochs * len(train_data)))
                set_lr(lr_ratio)
    return log


def setup_dataset(dataset, num_labels, dataset_name, num_examples=None):
    np.random.seed(0)
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if num_examples == None:
        num_examples = len(dataset)

    if dataset_name in ('mnli', 'hans'):
        num_per_label = num_examples // num_labels
        current_num_per_label = [0 for _ in range(num_labels)]
        examples = []
        for i in idx:
            exmp = dataset[int(i)]
            if all([num == num_per_label for num in current_num_per_label]):
                break
            if current_num_per_label[exmp['label']] < num_per_label:
                current_num_per_label[exmp['label']] += 1
                examples.append(((exmp['hypothesis'], exmp['premise']), exmp['label']))
    elif dataset_name in ('cola', 'sst2'):
        num_per_label = num_examples // num_labels
        current_num_per_label = [0 for _ in range(num_labels)]
        examples = []
        for i in idx:
            exmp = dataset[int(i)]
            if all([num == num_per_label for num in current_num_per_label]):
                break
            if current_num_per_label[exmp['label']] < num_per_label:
                current_num_per_label[exmp['label']] += 1
                examples.append((exmp['sentence'], exmp['label']))
    else:
        assert False, "Not implemented for dataset {}".format(dataset_name)
    return examples


def convert_to_mlm(examples):
    classes = [['yes', 'right'], ['maybe'], ['wrong', 'no']]
    mlm_examples = []
    for (hyp, premise), label in examples:
        hyp = word_tokenize(hyp)
        premise = [classes[label][0]] + [','] + word_tokenize(premise)
        mask1 = [0 for _ in hyp]
        mask2 = [0 for _ in premise]
        mask2[0] = 1
        mlm_examples.append((((hyp, premise), (mask1, mask2)), label))
    return mlm_examples, classes


if __name__ == "__main__":

    args = parser.parse_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    sanity = args.sanity
    do_mlm = args.mlm
    model_type = args.model
    reload = args.reload
    data_size = args.data_size
    eval_data_size = args.eval_data_size
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    check_every = args.check_every
    initial_check = args.initial_check
    xp_dir = args.xp_dir
    output_dir = os.path.join(xp_dir, f"{model_type}_{'pet' if do_mlm else 'finetuned'}")
    plotting = args.plotting

    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    mnli_dataset = nlp.load_dataset('glue', 'mnli')
    hans_dataset = nlp.load_dataset('hans', split="validation")
    num_labels_mnli = max([exmp['label'] for exmp in mnli_dataset['validation_matched']]) + 1
    num_labels_hans = max([exmp['label'] for exmp in hans_dataset]) + 1

    if sanity:
        data_size = 100
        eval_data_size = 100

    train_data = setup_dataset(mnli_dataset['train'], num_labels_mnli, 'mnli', data_size)
    dev_data = setup_dataset(mnli_dataset['validation_matched'], num_labels_mnli, 'mnli', eval_data_size)
    test_data = setup_dataset(mnli_dataset['validation_mismatched'], num_labels_mnli, 'mnli', eval_data_size)
    hans_data = setup_dataset(hans_dataset, num_labels_hans, 'hans', eval_data_size)

    if do_mlm:
        train_data, classes = convert_to_mlm(train_data)
        dev_data, classes = convert_to_mlm(dev_data)
        test_data, classes = convert_to_mlm(test_data)
        hans_data, hans_classes = convert_to_mlm(hans_data)

    print(len(train_data), len(dev_data), len(test_data))
    print(train_data[0], dev_data[0], test_data[0])

    if model_type == "bert":
        lm = ModelWrapper('bert', 'bert-base-uncased')
    elif model_type == "roberta":
        lm = ModelWrapper('roberta', 'roberta-base')
    else:
        raise KeyError(f"model type {model_type} not supported")
    if do_mlm:
        model = MLMClassifier(lm, classes)
    else:
        model = MLPClassifier(lm, num_labels_mnli)

    if not reload:
        log = train(model, train_data, dev_data, hans_data, output_dir=output_dir, verbose=True, epochs=epochs,
                    batch_size=train_batch_size, eval_batch_size=eval_batch_size, check_every=check_every,
                    initial_check=initial_check)

    if plotting:
        for key in log[0].keys():
            plt.plot(np.arange(len(log)), [a[key] for a in log], color='blue')
            plt.title(key)
            plt.show()

    print("reloading model")
    model = torch.load(os.path.join(output_dir, f"best_{model.model_type}"))
    dev_acc = evaluate(model, dev_data, eval_batch_size)
    test_acc = evaluate(model, test_data, eval_batch_size)
    hans_acc = evaluate(model, hans_data, eval_batch_size, hans=True)
    log.append({'dev_acc': dev_acc, 'test_acc': test_acc, 'hans_acc': hans_acc})
    print("Final results: {}".format(log[-1]))
    json.dump(log, open(os.path.join(output_dir, "log.json"), 'w'), ensure_ascii=False, indent=2)

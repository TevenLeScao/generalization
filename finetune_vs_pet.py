import json
import logging
import os
from functools import partial

from matplotlib import pyplot as plt
from tqdm import tqdm
import nlp
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from argparsing import parser
from classifiers import MLPClassifier, MLMClassifier
from model_wrapper import ModelWrapper
from utils import add_period

logger = logging.getLogger(__name__)


def evaluate(model, eval_data, subbatch_size=64, hans=False):
    model.eval()
    with torch.no_grad():
        preds = None
        for j in tqdm(range(0, len(eval_data), subbatch_size)):
            examples = eval_data[j:j + subbatch_size]
            logits = model(examples["premise"], examples["hypothesis"])
            preds_batch = np.argmax(logits.cpu().numpy(), axis=1)
            if preds is None:
                preds = preds_batch
            else:
                preds = np.concatenate((preds, preds_batch), axis=0)
            if hans:
                preds = np.clip(preds, 0, 1)
        eval_acc = np.sum(np.array([exmp['label'] for exmp in eval_data]) == preds) / len(eval_data)
    return eval_acc


def train(model, train_data, dev_data, hans_easy_data, hans_hard_data, output_dir, lr_base=3e-5, lr_warmup_frac=0.1,
          epochs=5, batch_size=32, subbatch_size=8, eval_batch_size=64, check_every=2048, initial_check=False,
          verbose=True):
    print("lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}, len(train_data): {}".format(
        lr_base, lr_warmup_frac, epochs, batch_size, len(train_data)))

    params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    optimizer = torch.optim.Adam([
        {'params': params, 'lr': 0., 'lr_base': lr_base, 'name': model.model_type}, ], lr=0.)

    def set_lr(lr_ratio):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    log = []
    processed = 0
    if initial_check:
        check_processed = check_every
    else:
        check_processed = 0
    train_acc_sum, train_acc_n = 0, 0
    best_dev_acc = 0
    step = 0

    for epoch in tqdm(range(epochs)):
        train_data.shuffle()
        for i in tqdm(range(0, len(train_data), batch_size)):
            examples = train_data[i:i + batch_size]
            model.train()
            optimizer.zero_grad()

            if check_processed >= check_every:
                dev_acc = evaluate(model, dev_data, eval_batch_size)
                hans_easy_acc = evaluate(model, hans_easy_data, eval_batch_size, hans=True)
                hans_hard_acc = evaluate(model, hans_hard_data, eval_batch_size, hans=True)
                train_acc = train_acc_sum / train_acc_n if train_acc_n > 0 else None
                log.append({'dev_acc': dev_acc,
                            'hans_easy_acc': hans_easy_acc,
                            'hans_hard_acc': hans_hard_acc,
                            'total_hans_acc': (hans_easy_acc + hans_hard_acc) / 2,
                            'train_acc': train_acc})
                if local_rank == -1 or torch.distributed.get_rank() == 0 and not sanity:
                    wandb.log({'dev_acc': dev_acc,
                               'hans_easy_acc': hans_easy_acc,
                               'hans_hard_acc': hans_hard_acc,
                               'total_hans_acc': (hans_easy_acc + hans_hard_acc) / 2,
                               'train_acc': train_acc},
                              step=step)
                train_acc_sum, train_acc_n = 0, 0
                check_processed -= check_every
                if verbose:
                    print("Epoch: {}, Log: {}".format(epoch, log[-1]))
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(model, os.path.join(output_dir, f"best_{model.model_type}"))

            for j in range(0, len(examples), subbatch_size):
                examples_subbatch = {k: v[j:j + subbatch_size] for k, v in examples.items()}

                # compute loss, also log other metrics
                logits = model(examples_subbatch["premise"], examples_subbatch["hypothesis"])
                labels = torch.tensor(examples_subbatch["label"], device=logits.device)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                del loss

                batch_acc = (logits.argmax(axis=1) == labels).sum().item() / len(labels)
                train_acc_sum += batch_acc
                train_acc_n += 1

            optimizer.step()
            step += 1
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


def add_prototype(example, mask_token):
    example["hypothesis"] = f"{mask_token}, " + example["hypothesis"]
    return example


def setup_device(local_rank):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif local_rank == -1:
        # if n_gpu is > 1 we'll use nn.DataParallel.
        # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
        # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
        # trigger an error that a device index is missing. Index 0 takes into account the
        # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
        # will use the first GPU in that env, i.e. GPU#1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # Here, we'll use torch.distributed.
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        n_gpu = 1

    if device.type == "cuda":
        torch.cuda.set_device(device)

    return device, n_gpu


def run_name(model_type, mlm, samples):
    return f'{"mlm" if mlm else "finetuned"}_{model_type}_{samples}-datapts'


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
    seed = args.seed
    local_rank = args.local_rank

    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    device, n_gpu = setup_device(local_rank)
    mnli_dataset = nlp.load_dataset('glue', 'mnli')
    train_data = mnli_dataset['train']
    dev_data = mnli_dataset['validation_matched']
    test_data = mnli_dataset['validation_mismatched']
    hans_easy_data = nlp.load_dataset('hans', split="validation").filter(lambda x: x['label'] == 0)
    hans_hard_data = nlp.load_dataset('hans', split="validation").filter(lambda x: x['label'] == 1)
    num_labels_mnli = 3

    if model_type == "bert":
        lm = ModelWrapper('bert', 'bert-base-uncased', device=device)
    elif model_type == "roberta":
        lm = ModelWrapper('roberta', 'roberta-base', device=device)
    else:
        raise KeyError(f"model type {model_type} not supported")
    if do_mlm:
        classes = [['yes', 'right'], ['maybe'], ['wrong', 'no']]
        model = MLMClassifier(lm, classes, device=device).to(device)
    else:
        model = MLPClassifier(lm, num_labels_mnli, device=device).to(device)

    if sanity:
        data_size = 100
        eval_data_size = 100

    if data_size is not None:
        train_data = train_data.select(list(range(min(len(train_data), data_size))))
    if eval_data_size is not None:
        dev_data = dev_data.select(list(range(min(len(dev_data), eval_data_size))))
        hans_easy_data = hans_easy_data.select(list(range(min(len(hans_easy_data), eval_data_size))))
        hans_hard_data = hans_hard_data.select(list(range(min(len(hans_hard_data), eval_data_size))))

    if do_mlm:
        train_data = train_data.map(partial(add_prototype, mask_token=lm.tokenizer.mask_token))
        dev_data = dev_data.map(partial(add_prototype, mask_token=lm.tokenizer.mask_token))
        test_data = test_data.map(partial(add_prototype, mask_token=lm.tokenizer.mask_token))
        hans_easy_data = hans_easy_data.map(partial(add_prototype, mask_token=lm.tokenizer.mask_token))
        hans_hard_data = hans_hard_data.map(partial(add_prototype, mask_token=lm.tokenizer.mask_token))

    print(len(train_data), len(dev_data), len(test_data))
    print(train_data[0], dev_data[0], test_data[0])

    if (local_rank == -1 or torch.distributed.get_rank() == 0) and not sanity:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "huggingface"), name=run_name(model_type, do_mlm, len(train_data))
        )

    if not reload:
        log = train(model, train_data, dev_data, hans_easy_data, hans_hard_data, output_dir=output_dir, verbose=True,
                    epochs=epochs, batch_size=train_batch_size, eval_batch_size=eval_batch_size,
                    check_every=check_every, initial_check=initial_check)

    if plotting:
        for key in log[0].keys():
            plt.plot(np.arange(len(log)), [a[key] for a in log], color='blue')
            plt.title(key)
            plt.show()

    print("reloading model")
    model = torch.load(os.path.join(output_dir, f"best_{model.model_type}"))
    dev_acc = evaluate(model, dev_data, eval_batch_size)
    test_acc = evaluate(model, test_data, eval_batch_size)
    hans_easy_acc = evaluate(model, hans_easy_data, eval_batch_size, hans=True)
    hans_hard_acc = evaluate(model, hans_hard_data, eval_batch_size, hans=True)
    log.append(
        {'dev_acc': dev_acc, 'test_acc': test_acc, 'hans_easy_acc': hans_easy_acc, 'hans_hard_acc': hans_hard_acc,
         'total_hans_acc': (hans_easy_acc + hans_hard_acc) / 2, })
    print("Final results: {}".format(log[-1]))
    json.dump(log, open(os.path.join(output_dir, "log.json"), 'w'), ensure_ascii=False, indent=2)

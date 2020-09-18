import json
import logging
import os
import random
from functools import partial
from contextlib import contextmanager

from matplotlib import pyplot as plt
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from tqdm import tqdm
import nlp
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from transformers.trainer import SequentialDistributedSampler

from argparsing import parser
from classifiers import MLPClassifier, MLMClassifier
from model_wrapper import ModelWrapper
from utils import add_period, distributed_broadcast_scalars

logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    print("\n\n\nbarrier\n\n\n")
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def evaluate(model, eval_data, hans=False, shots=None, train_data=None):
    model.eval()
    with torch.no_grad():
        preds = None
        for examples in tqdm(eval_data):
            if shots is None:
                logits = model(examples["premise"], examples["hypothesis"])
            else:
                if shots > 0:
                    assert train_data is not None, "shots passed, indicating lm-style inference, but no train_data"
                    shot_indexes = random.sample(list(range(len(train_data))), shots)
                    shot_data = train_data[shot_indexes]
                    logits = model(examples["premise"], examples["hypothesis"], shot_data["premise"],
                                   shot_data["labeled_hypothesis"])
                else:
                    logits = model(examples["premise"], examples["hypothesis"], [], [])
            preds_batch = np.argmax(logits.cpu().numpy(), axis=1)
            if preds is None:
                preds = preds_batch
            else:
                preds = np.concatenate((preds, preds_batch), axis=0)
            if hans:
                preds = np.clip(preds, 0, 1)
        eval_acc = np.array(torch.cat([exmp['label'] for exmp in eval_data])) == preds
    return eval_acc


def train(model, train_data, dev_data, hans_easy_data, hans_hard_data, output_dir, lr_base=3e-5, lr_warmup_frac=0.1,
          epochs=5, subbatch_size=8, check_every=2048, initial_check=False,
          shots=None, verbose=True, model_type=None):
    print("lr_base: {}, lr_warmup_frac: {}, epochs: {}, len(train_data): {}".format(
        lr_base, lr_warmup_frac, epochs, len(train_data)))

    params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    optimizer = torch.optim.Adam([
        {'params': params, 'lr': 0., 'lr_base': lr_base, 'name': model_type}, ], lr=0.)

    def set_lr(lr_ratio):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    log = []
    processed = 0
    if initial_check:
        check_processed = check_every
    else:
        check_processed = 0
    train_acc = []
    best_dev_acc = 0
    step = 0

    for epoch in tqdm(range(epochs)):
        for examples in tqdm(train_data):
            model.train()
            optimizer.zero_grad()

            if check_processed >= check_every:
                dev_acc = evaluate(model, dev_data, eval_batch_size, shots=shots, train_data=train_data)
                print(dev_acc)
                print(dev_acc.shape)
                hans_easy_acc = evaluate(model, hans_easy_data, hans=True, shots=shots,
                                         train_data=train_data)
                print(hans_easy_acc)
                print(hans_easy_acc.shape)
                hans_hard_acc = evaluate(model, hans_hard_data, hans=True, shots=shots,
                                         train_data=train_data)
                print(hans_hard_acc)
                print(hans_hard_acc.shape)
                if local_rank != -1:
                    dev_acc = distributed_broadcast_scalars(dev_acc).cpu().mean().item()
                    hans_easy_acc = distributed_broadcast_scalars(hans_easy_acc).cpu().mean().item()
                    hans_hard_acc = distributed_broadcast_scalars(hans_hard_acc).cpu().mean().item()
                    if train_acc:
                        train_acc = distributed_broadcast_scalars(train_acc).cpu().mean().item()
                else:
                    dev_acc = np.mean(dev_acc)
                    hans_easy_acc = np.mean(hans_easy_acc)
                    hans_hard_acc = np.mean(hans_hard_acc)
                    train_acc = np.mean(train_acc)
                    log.append({'dev_acc': dev_acc,
                                'hans_easy_acc': hans_easy_acc,
                                'hans_hard_acc': hans_hard_acc,
                                'total_hans_acc': (hans_easy_acc + hans_hard_acc) / 2,
                                'train_acc': train_acc})
                if (local_rank == -1 or torch.distributed.get_rank() == 0):
                    if not sanity:
                        wandb.log({'dev_acc': dev_acc,
                                   'hans_easy_acc': hans_easy_acc,
                                   'hans_hard_acc': hans_hard_acc,
                                   'total_hans_acc': (hans_easy_acc + hans_hard_acc) / 2,
                                   'train_acc': train_acc},
                                  step=step)
                        if dev_acc > best_dev_acc:
                            best_dev_acc = dev_acc
                            torch.save(model, os.path.join(output_dir, f"best_{model_type}"))
                if verbose:
                    print("Epoch: {}, Log: {}".format(epoch, log[-1]))
                train_acc = []
                check_processed -= check_every

            for j in range(0, len(examples), subbatch_size):
                examples_subbatch = {k: v[j:j + subbatch_size] for k, v in examples.items()}

                # compute loss, also log other metrics
                logits = model(examples_subbatch["premise"], examples_subbatch["hypothesis"])
                labels = examples_subbatch["label"].to(device=logits.device)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                del loss

                batch_acc = (logits.argmax(axis=1) == labels).sum().item() / len(labels)
                train_acc.append(batch_acc)

            optimizer.step()
            step += 1
            processed += len(examples)
            check_processed += len(examples)

            # warmup from 0 to lr_base for lr_warmup_frac
            lr_ratio = min(1, processed / (lr_warmup_frac * epochs * len(train_data)))
            set_lr(lr_ratio)
    return log


def add_prototype(example, mask_token, classes):
    example["premise"] = add_period(example["premise"])
    fill_in = random.choice(classes[example['label']])
    example["labeled_hypothesis"] = f"{fill_in}, " + example["hypothesis"]
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


def run_name(model_type, mlm, shots, samples):
    if mlm and shots is not None:
        type = "prompts"
        samples = shots
    elif mlm and shots is None:
        type = "mlm"
    else:
        type = "finetuned"
    return f'{type}_{model_type}_{samples}-datapts'


if __name__ == "__main__":

    args = parser.parse_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    # args that appear several times
    sanity = args.sanity
    do_mlm = args.mlm
    shots = args.shots
    token_limit = args.token_limit
    model_type = args.model
    data_size = args.data_size
    eval_data_size = args.eval_data_size
    eval_batch_size = args.eval_batch_size
    train_batch_size = args.train_batch_size
    local_rank = args.local_rank

    device, n_gpu = setup_device(local_rank)
    if model_type == "bert":
        lm = ModelWrapper('bert', 'bert-base-uncased', token_limit=token_limit, device=device)
    elif model_type == "roberta":
        lm = ModelWrapper('roberta', 'roberta-base', token_limit=token_limit, device=device)
    else:
        raise KeyError(f"model type {model_type} not supported")
    if do_mlm:
        classes = [['yes', 'right'], ['maybe'], ['wrong', 'no']]
        model = MLMClassifier(lm, classes, shots=shots, token_limit=token_limit, device=device).to(device)
    else:
        # MNLI has 3 labels
        model = MLPClassifier(lm, 3, device=device).to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    mnli_dataset = nlp.load_dataset('glue', 'mnli')
    train_data = mnli_dataset['train']
    dev_data = mnli_dataset['validation_matched']
    test_data = mnli_dataset['validation_mismatched']
    hans_easy_data = nlp.load_dataset('hans', split="validation").filter(lambda x: x['label'] == 0)
    hans_hard_data = nlp.load_dataset('hans', split="validation").filter(lambda x: x['label'] == 1)

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
        mask_token = lm.tokenizer.mask_token
        train_data = train_data.map(partial(add_prototype, mask_token=mask_token, classes=classes))
        dev_data = dev_data.map(partial(add_prototype, mask_token=mask_token, classes=classes))
        test_data = test_data.map(partial(add_prototype, mask_token=mask_token, classes=classes))
        hans_easy_data = hans_easy_data.map(partial(add_prototype, mask_token=mask_token, classes=classes))
        hans_hard_data = hans_hard_data.map(partial(add_prototype, mask_token=mask_token, classes=classes))

    train_sampler = RandomSampler if local_rank == -1 else DistributedSampler
    eval_sampler = RandomSampler if local_rank == -1 else SequentialDistributedSampler
    total_batch_size = args.accumulate * train_batch_size
    train_data = DataLoader(train_data, sampler=train_sampler(train_data), batch_size=total_batch_size)
    dev_data = DataLoader(dev_data, sampler=eval_sampler(dev_data), batch_size=eval_batch_size)
    test_data = DataLoader(test_data, sampler=eval_sampler(test_data), batch_size=eval_batch_size)
    hans_easy_data = DataLoader(hans_easy_data, sampler=eval_sampler(hans_easy_data), batch_size=eval_batch_size)
    hans_hard_data = DataLoader(hans_hard_data, sampler=eval_sampler(hans_hard_data), batch_size=eval_batch_size)

    name = run_name(model_type, do_mlm, shots, len(train_data))
    output_dir = os.path.join(args.xp_dir, f"{model_type}_{'pet' if do_mlm else 'finetuned'}")
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    if (local_rank == -1 or torch.distributed.get_rank() == 0) and not sanity:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "huggingface"), name=name
        )

    if not args.reload:
        log = train(model, train_data, dev_data, hans_easy_data, hans_hard_data, output_dir=output_dir, verbose=True,
                    epochs=args.epochs, subbatch_size=args.train_batch_size,
                    check_every=args.check_every, initial_check=args.initial_check,
                    shots=shots, model_type=model_type)

    if args.plotting:
        for key in log[0].keys():
            plt.plot(np.arange(len(log)), [a[key] for a in log], color='blue')
            plt.title(key)
            plt.show()

    if args.epochs > 0 and not sanity:
        print("reloading model")
        model = torch.load(os.path.join(output_dir, f"best_{model_type}"))
    dev_acc = np.mean(evaluate(model, dev_data, eval_batch_size, shots=shots, train_data=train_data))
    test_acc = np.mean(evaluate(model, test_data, eval_batch_size, shots=shots, train_data=train_data))
    hans_easy_acc = np.mean(
        evaluate(model, hans_easy_data, hans=True, shots=shots, train_data=train_data))
    hans_hard_acc = np.mean(
        evaluate(model, hans_hard_data, hans=True, shots=shots, train_data=train_data))
    log.append(
        {'dev_acc': dev_acc, 'test_acc': test_acc, 'hans_easy_acc': hans_easy_acc, 'hans_hard_acc': hans_hard_acc,
         'total_hans_acc': (hans_easy_acc + hans_hard_acc) / 2, })
    print("Final results: {}".format(log[-1]))
    json.dump(log, open(os.path.join(output_dir, "log.json"), 'w'), ensure_ascii=False, indent=2)

from collections import OrderedDict

import torch
from torch.utils.data import RandomSampler, DistributedSampler


def add_period(sentence):
    if sentence[-1] == ".":
        return sentence
    else:
        return sentence + "."


def combine_encodings(previous, current):
    if previous is None:
        return current
    else:
        return OrderedDict((k, v + current[k]) for k, v in previous.items())


def distributed_concat(tensor: "torch.Tensor", num_total_examples=None) -> "torch.Tensor":
    try:
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def distributed_broadcast_scalars(scalars, num_total_examples=None):
    try:
        tensorized_scalar = torch.Tensor(scalars).cuda()
        output_tensors = [tensorized_scalar.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensorized_scalar)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")

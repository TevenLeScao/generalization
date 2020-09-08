import torch


def add_period(sentence):
    if sentence[-1] == ".":
        return sentence
    else:
        return sentence + "."


def combine_encodings(previous, current):
    if previous is None:
        return current
    else:
        return {k: torch.stack((v, current[k]), dim=1) for k, v in previous.items()}

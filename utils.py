from collections import OrderedDict

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
        return OrderedDict((k, v + current[k]) for k, v in previous.items())

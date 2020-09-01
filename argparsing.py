import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mlm", action="store_true")
parser.add_argument("--data_size", type=int, default=10000)
parser.add_argument("--model", type=str, default=None, choices=["bert", "roberta"])

# parser.add_argument("--sanity", action="store_true")
# parser.add_argument("--depth", type=int, default=2)
# parser.add_argument("--width", type=int, default=64)
# parser.add_argument("--inner", type=int, default=None)
# parser.add_argument("--batch_size", type=int, default=128)
# parser.add_argument("--accum", type=int, default=32)
# parser.add_argument("--warmup", type=int, default=100)
# parser.add_argument("--lr", type=float, default=0.0025)
# parser.add_argument("--log_freq", type=int, default=100)
# parser.add_argument("--suffix", type=str, default=None)
# parser.add_argument("--local_rank", type=int, default=-1)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sanity", action="store_true")
parser.add_argument("--mlm", action="store_true")
parser.add_argument("--shots", type=int, default=None)
parser.add_argument("--model", type=str, default="bert", choices=["bert", "roberta"])
parser.add_argument("--reload", action="store_true")
parser.add_argument("--data_size", type=int, default=None)
parser.add_argument("--eval_data_size", type=int, default=None)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--token_limit", type=int, default=None)
parser.add_argument("--check_every", type=int, default=2048)
parser.add_argument("--initial_check", action="store_true")
parser.add_argument("--xp_dir", type=str, default="experiments")
parser.add_argument("--plotting", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)

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

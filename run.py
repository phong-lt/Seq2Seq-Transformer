from config.config import get_config
from core.executing import Executor
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Seq2Seq Args')

    parser.add_argument("--mode", choices=['train', 'val'],
                      help='{train, val}',
                      type=str, required=True)
    
    parser.add_argument("--config-file", type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    config = get_config(args.config_file)

    exec = Executor(config)

    exec.run(args.mode)
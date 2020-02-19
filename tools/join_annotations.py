import json
from glob import glob

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('dataset_1', help='dataset_1')
    parser.add_argument('dataset_2', help='dataset_2')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open(args.dataset_1) as json_file:
        dataset1 = json.load(json_file)
    with open(args.dataset_2) as json_file:
        dataset2 = json.load(json_file)



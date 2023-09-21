import argparse
import json

import tensorflow as tf

from benchmarking.utils import set_random_seed, generate_datasets, tf_cap_memory


def main(args):

    config = json.load(open(args.config))
    dataset_configs = json.load(open(args.dataset_configs))
    dataset_name = args.dataset_name

    # select specific dataset
    if dataset_name:
        dataset_configs = {dataset_name: dataset_configs[dataset_name]}

    # set random seed
    if 'seed' in config:
        set_random_seed(config['seed'])

    for ds_name, ds_config in dataset_configs.items():
        generate_datasets(config, ds_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmented dataset')
    parser.add_argument('--dataset_configs', '-d',
                        help='path to dataset configs',
                        default='config/datasets.json')
    parser.add_argument('--config', '-c', help='path to generate ds config',
                        default='configs/generate_dataset.json')
    parser.add_argument(
        '--dataset_name', help='optional - specify a dataset')
    args = parser.parse_args()
    main(args)

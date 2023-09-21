import argparse
import json
import copy
import os

from benchmarking.utils import set_random_seed, tf_cap_memory
from benchmarking.train_experiment import TrainExperiment


def main(args):

    # select gpus
    if args.gpus:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

    # extract configs from args
    config = json.load(open(args.config))
    model_configs = json.load(open(args.model_configs))
    dataset_configs = json.load(open(args.dataset_configs))
    vec_configs = json.load(open(args.vec_configs))
    model_name = args.model_name
    dataset_name = args.dataset_name
    vec_name = args.vec_name

    # select shards
    if model_name:
        model_configs = {model_name: model_configs[model_name]}
    if dataset_name:
        dataset_configs = {dataset_name: dataset_configs[dataset_name]}
    if vec_name:
        vec_configs = {vec_name: vec_configs[vec_name]}

    # set random seed
    if 'seed' in config:
        set_random_seed(config['seed'])

    if args.exp_dir:
        config['base_dir'] = args.exp_dir

    tf_cap_memory()

    # generate and run all experiments
    for d_name, d_config in dataset_configs.items():
        for m_name, m_config in model_configs.items():

            # select the vectorizers we want to run out of the list
            vec_names = list(vec_configs.keys())
            vec_start_idx = args.vec_start_idx
            vec_end_idx = args.vec_end_idx if args.vec_end_idx else len(vec_names)
            vec_names = vec_names[vec_start_idx:vec_end_idx]

            for v_name in vec_names:
                v_config = vec_configs[v_name]
                config = copy.copy(config)
                config['model'] = m_config
                config['dataset'] = d_config
                config['vectorizer'] = v_config

                # run and save experiment
                experiment = TrainExperiment(config)
                experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train text models')

    # model, dataset, and vectorizer configs will contain multiple configs.
    # the user can specify each one, if desired, or the trainer will generate
    # training jobs for each one in the configs. i.e. if model_name is not
    # selected, train all models in model_configs
    parser.add_argument('--config', '-c', help='path to training config',
                        default='configs/train_classification.json.json')
    parser.add_argument('--model_configs', '-m', help='path to model configs',
                        default='configs/models.json')
    parser.add_argument('--dataset_configs', '-d',
                        help='path to dataset configs',
                        default='configs/datasets.json')
    parser.add_argument('--vec_configs', '-v',
                        help='path to vectorizer configs',
                        default='configs/vectorizers.json')
    parser.add_argument('--exp_dir',
                        help='override default experiment dir')
    parser.add_argument(
        '--model_name', help='specify a model from model_configs')
    parser.add_argument(
        '--dataset_name', help='specify a dataset from dataset_configs')
    parser.add_argument(
        '--vec_name', help='specify an vectorizer from vec_configs')
    parser.add_argument(
        "--vec_start_idx",
        "-s",
        type=int,
        help="start idx in sorted vectorizer configs",
        default=0,
    )
    parser.add_argument(
        "--vec_end_idx", "-e", type=int, help="end idx in sorted vectorizer configs (zero-indexed, exclusive)"
    )
    parser.add_argument('--gpus', help='list of gpus', default='')

    args = parser.parse_args()
    main(args)

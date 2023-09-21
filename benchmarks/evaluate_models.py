import argparse
import json
import tensorflow as tf
from pathlib import Path

from benchmarking.utils import set_random_seed, tf_cap_memory
from benchmarking.eval_experiment import EvalExperiment


def main(args):
    tf_cap_memory()

    # extract configs from args
    config = json.load(open(args.config))

    for eval_config in config["evaluations"]:
        dataset_configs = json.load(open(args.dataset_configs))
        dataset_name = eval_config['dataset_name']

        # select specific dataset
        if dataset_name:
            dataset_configs = {dataset_name: dataset_configs[dataset_name]}

        # set random seed
        if 'seed' in eval_config:
            set_random_seed(eval_config['seed'])

        experiment = EvalExperiment(config=eval_config)

        for ds_name, ds_config in dataset_configs.items():
            model_path = eval_config["model_path"]

            # single model
            if tf.saved_model.contains_saved_model(model_path):
                print("Evaluating model at %s" % model_path)
                experiment.evaluate(model_path, ds_config)
                continue

            # otherwise model_path is a directory of models
            p = Path(model_path)
            subdirectories = [x for x in p.iterdir() if x.is_dir()]

            for subdir in subdirectories:
                model_path = str(subdir)
                if tf.saved_model.contains_saved_model(model_path):
                    print("Evaluating model at %s" % model_path)
                    experiment.evaluate(model_path, ds_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train text models')
    parser.add_argument('--config', '-c', help='path to evaluation config',
                        default='config/evaluate.json')
    parser.add_argument('--dataset_configs', '-d',
                        help='path to dataset configs',
                        default='config/datasets.json')

    args = parser.parse_args()
    main(args)

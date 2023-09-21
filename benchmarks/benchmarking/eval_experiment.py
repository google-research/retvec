import os
import json
from copy import copy
import numpy as np
import time

import tensorflow as tf

from typing import Dict
from .utils import generate_datasets, count_model_params, get_flops
from pathlib import Path
import tensorflow_addons as tfa

from .vectorizers import get_vectorizer
from .preprocessors import preprocess_text
from .datasets import Datasets
from .optimizers import *


class EvalExperiment:

    def __init__(self, config) -> None:
        self.config = config

    def evaluate(self, model_path):
        """Evaluate the model at model_path on the eval datasets
        given by dataset config. If the eval datasets have not been
        generated, generates the datasets first then does the evaluation.
        """
        config = self.config
        results_dir, exp_name = self._get_results_dir(model_path)
        results_file = results_dir + '/eval_results.json'

        if os.path.exists(results_file):
            os.remove(results_file)

        train_config = self._get_train_config(model_path)
        dataset_config = train_config['dataset']

        # generate datasets if they do not exist
        generate_datasets(config, dataset_config)

        dataset_name = dataset_config['name']

        datasets_dir = config["datasets_dir"]
        dataset_dir = datasets_dir + "%s-v%s" % (
            dataset_name, dataset_config['version'])

        # load model
        print("Loading SavedModel")
        saved_model = tf.keras.models.load_model(model_path)
        saved_model.summary()

        # count model params and FLOPs
        trainable, nontrainable, total = count_model_params(saved_model)
        flops = get_flops(saved_model)

        print(config)
        aug_config = config["augmentations"]
        for aug_type in aug_config['type']:
            for bsize in aug_config['block_size']:
                for pct in aug_config['pct_words_to_swap']:

                    # get the aug dataset and evaluate model on it
                    dataset_path = dataset_dir + \
                        '/%s_%s_%s.npz' % (aug_type, bsize, pct)
                    print(f"Dataset at {dataset_path}")

                    results = {}

                    # add eval parameters to results
                    results["model_path"] = model_path
                    results["aug_type"] = aug_type
                    results["aug_block_size"] = bsize
                    results["aug_percent"] = pct
                    results["exp_name"] = exp_name
                    results["dataset_name"] = dataset_name
                    results["dataset_version"] = dataset_config['version']
                    results["params_trainable"] = int(trainable)
                    results["params_nontrainable"] = int(nontrainable)
                    results["params_total"] = int(total)
                    results["flops"] = flops

                    results['model_name'] = train_config['model']['name']
                    results['model_type'] = train_config['model']['type']
                    results['vec_name'] = train_config['vectorizer']['name']
                    results['vec_type'] = train_config['vectorizer']['type']

                    results = self._evaluate_model(
                        saved_model, dataset_path, train_config, results)

                    with open(results_file, 'a+') as f:
                        json.dump(results, f)
                        f.write('\n')

    def _evaluate_model(self,
                        saved_model: tf.keras.Model,
                        dataset_path: str,
                        train_config: Dict,
                        results: Dict = {}) -> Dict:
        npzfile = np.load(dataset_path, allow_pickle=True)
        x = npzfile['x']
        y = npzfile['y']
        x = tf.constant(x, tf.string)
        y = tf.constant(y)

        # insert CLS token for BERT models if neccessary
        if train_config['model']['type'] == 'bert':
            if train_config['vectorizer']['type'] not in ['byte_level_bpe', 'sentencepiece']:
                datasets = Datasets()
                x, y = datasets._preprocess_insert_cls(x, y)

        vectorizer, tokenizer = get_vectorizer(x, train_config)

        # preprocess
        preprocess_start = time.time()
        x = preprocess_text(x, vectorizer, tokenizer, train_config)

        results['preprocess_time'] = time.time() - preprocess_start

        eval_start = time.time()
        output = saved_model.evaluate(
            x, y, batch_size=self.config["batch_size"])
        results['eval_time'] = time.time() - eval_start

        metrics = saved_model.metrics_names
        for i in range(len(output)):
            results[metrics[i]] = output[i]
        return results

    def _get_results_dir(self, model_path: str):
        """Get results file based on model path, since you can extract
        the experiment name from the model path."""
        # FIXME (marinazh): this is too fragile, currently dependent on how the
        # train experiment results are structured
        # model path in the form of models_dir/models/exp_name
        p = Path(model_path)
        exp_name = p.parts[-1]
        models_dir = p.parent.parent
        results_dir = os.path.join(models_dir, 'results', exp_name)
        return results_dir, exp_name

    def _get_train_config(self, model_path: str):
        results_dir, exp_name = self._get_results_dir(model_path)
        train_file = results_dir + '/train_config.json'

        train_config = json.load(open(train_file))
        return train_config

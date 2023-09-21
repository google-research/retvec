import json
import os
import random
import tempfile
import time
from typing import Tuple
import fasttext
from tqdm.auto import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

from .augmenter import AUG_TYPE_TO_AUGMENTER
from .datasets import Datasets


def tf_cap_memory():
    "Avoid TF to hog memory before needing it"
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def set_random_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def generate_datasets(config, dataset_config):
    """Generates augmented datasets based on a (generate dataset)
    config and a dataset config."""

    def _create_dataset_dir():
        datasets_dir = config["datasets_dir"]
        save_dir = datasets_dir + "%s-v%s" % (
            dataset_config['name'],
            dataset_config['version'])
        config_dir = save_dir + '/config'

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)

        # save configs in directory
        with open(config_dir + '/dataset.json', 'w') as f:
            json.dump(dataset_config, f)
        with open(config_dir + '/generate_dataset.json', 'w') as f:
            json.dump(config, f)

        return save_dir

    # dataset loader
    datasets = Datasets()

    # create save directory for dataset
    save_dir = _create_dataset_dir()

    aug_config = config['augmentations']
    text, labels = datasets.get(
        dataset_config['name'],
        datasets=dataset_config.get('datasets'),
        split='test')

    text = text.numpy()
    labels = labels.numpy()

    for aug_type in aug_config['type']:
        augmenter = AUG_TYPE_TO_AUGMENTER[aug_type](
            language=dataset_config['language'],
            min_len=aug_config['min_len'])

        for bsize in aug_config['block_size']:
            for pct in aug_config['pct_words_to_swap']:

                start_time = time.time()

                # save dataset in .npz format
                output_file = '%s_%s_%s.npz' % (aug_type, bsize, pct)
                filepath = os.path.join(save_dir, output_file)

                # check output csv path
                if os.path.exists(filepath):
                    if config["overwrite"]:
                        print("Preparing to overwrite %s" % filepath)
                        os.remove(filepath)
                    else:
                        print("Existing dataset at %s " % filepath)
                        continue

                aug_text, aug_labels = augmenter(
                    text, labels, bsize, pct,
                    aug_config['augmentations_per_example'])

                np.savez_compressed(filepath, x=aug_text, y=aug_labels)

                print(f"Dataset {output_file}: {time.time()-start_time}s")


def count_model_params(model: tf.keras.Model):
    trainable = np.sum([np.prod(v.shape)
                       for v in model.trainable_weights])
    nontrainable = np.sum([np.prod(v.shape)
                          for v in model.non_trainable_weights])
    total = trainable + nontrainable
    return trainable, nontrainable, total


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops

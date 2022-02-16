"""
 Copyright 2021 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
from time import time
from typing import Callable, Dict

import tensorflow as tf
from google.cloud import storage
from retvec.utils import get_outputs_info
from tensorflow import Tensor

from .constants import LANGUAGES

NUM_LANGUAGES = len(LANGUAGES)
COMPLEXITY_DIM = 5


@tf.function
def read_tfrecord(tfrecord: Tensor,
                  fixed_len: int = 16,
                  num_examples_per_token: int = 2) -> Dict[str, Tensor]:
    base_features = {
        'original_encoded': tf.io.FixedLenFeature([], tf.string),
        "index": tf.io.FixedLenFeature([], tf.int64),
    }
    record = []

    features = base_features.copy()
    for i in range(num_examples_per_token):
        features[f'aug_token{i}'] = tf.io.FixedLenFeature([], tf.string)
        features[f'aug_encoded{i}'] = tf.io.FixedLenFeature([], tf.string)

    rec = tf.io.parse_single_example(tfrecord, features)

    # convert numpy ndarrays to tensors
    rec['original_encoded'] = tf.io.parse_tensor(
        rec['original_encoded'], out_type=tf.float64)

    for i in range(num_examples_per_token):
        rec[f'aug_encoded{i}'] = tf.io.parse_tensor(
            rec[f'aug_encoded{i}'], out_type=tf.float64)

    # output a single record containing each augmented example
    record = {}
    prefixes = ['aug_token', 'aug_encoded']
    for p in prefixes:
        tensors = [rec[p + str(i)] for i in range(num_examples_per_token)]
        record[p] = tf.stack(tensors)

    for feature in base_features.keys():
        record[feature] = tf.stack([rec[feature]] * num_examples_per_token)

    # cast into float32 and int32
    record['index'] = tf.cast(record['index'], tf.int32)
    rec['original_encoded'] = tf.cast(record['original_encoded'], tf.float32)
    rec['aug_encoded'] = tf.cast(record['aug_encoded'], tf.float32)

    return record


def Sampler(shards_list: str,
            batch_size: int = 32,
            process_record: Callable = None,
            parallelism: int = tf.data.AUTOTUNE,
            file_parallelism: int = 1,
            prefetch_size: int = None,
            buffer_size: int = 10000,
            compression_type: str = "GZIP") -> tf.data.Dataset:

    total_shards = len(shards_list)
    print("found ", len(shards_list), 'shards', time())

    with tf.device('/cpu:0'):
        ds = tf.data.Dataset.from_tensor_slices(shards_list)
        ds = ds.shuffle(total_shards)

        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),  # noqa
            block_length=1,  # problem here is that we have non flat record
            num_parallel_calls=file_parallelism,
            cycle_length=file_parallelism,
            deterministic=False)

        ds = ds.map(read_tfrecord, num_parallel_calls=parallelism)
        ds = ds.shuffle(buffer_size)

        ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
        ds = ds.map(process_record, num_parallel_calls=parallelism)

        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch_size)
        return ds


def get_process_tfrecord_fn(outputs):
    """Return the transform to process the tfrecord
    and extract only the outputs in `outputs`.
    """

    @tf.function
    def process_tfrecord(e):
        x = {'token': e['aug_token']}
        y = {
            'ori_decoder': e['original_encoded'],
            'similarity': e['index'],
            'aug_decoder': e['aug_encoded'],
        }

        # FIXME: find a faster way to not parse all features
        # select only outputs we want
        y = {output: y[output] for output in outputs}
        return x, y

    return process_tfrecord


def get_dataset_samplers(bucket, path, config):
    core_count = os.cpu_count()
    client = storage.Client()

    loss, metrics, outputs = get_outputs_info(config)
    tfrecord_fn = get_process_tfrecord_fn(outputs)
    batch_size = config['batch_size']
    buffer_size = config['shuffle_buffer']

    train_files = []
    test_files = []

    for blob in client.list_blobs(bucket, prefix=path):
        if os.path.basename(str(blob.name)).startswith('train'):
            train_files.append(blob.name)

        if os.path.basename(str(blob.name)).startswith('test'):
            test_files.append(blob.name)

    train_shards = []
    test_shards = []

    for f in train_files:
        train_shards.append('gs://' + bucket + '/' + f)

    for f in test_files:
        test_shards.append('gs://' + bucket + '/' + f)

    train_ds = Sampler(train_shards,
                       process_record=tfrecord_fn,
                       batch_size=batch_size,
                       file_parallelism=core_count * 2,
                       parallelism=core_count,
                       buffer_size=buffer_size,
                       prefetch_size=10)

    test_ds = Sampler(test_shards,
                      process_record=tfrecord_fn,
                      file_parallelism=1,
                      batch_size=batch_size,
                      buffer_size=1000)

    return train_ds, test_ds

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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import tensorflow as tf
from google.cloud import storage
from tensorflow import Tensor
from tensorflow_similarity.losses import (
    CircleLoss,
    MultiSimilarityLoss,
    PNLoss,
    TripletLoss,
)

from ..layers import RETVecBinarizer


def read_tfrecord(
    tfrecord: Tensor, binarizer: RETVecBinarizer
) -> Dict[str, Tensor]:
    """Read TF record files for RETVec training datasets.

    Args:
        tfrecord: TF record input.

        binarizer: RETVecBinarizer used to encode words.
    """
    base_features = {
        "original_token": tf.io.FixedLenFeature([], tf.string),
        "idx": tf.io.FixedLenFeature([], tf.int64),
    }

    features = base_features.copy()

    for i in range(2):
        features["aug_token%s" % i] = tf.io.FixedLenFeature([], tf.string)
        features["aug_matrix%s" % i] = tf.io.FixedLenFeature([], tf.string)
        features["aug_vector%s" % i] = tf.io.FixedLenFeature([32], tf.int64)
        features["aug_bool%s" % i] = tf.io.FixedLenFeature([], tf.int64)

    rec = tf.io.parse_single_example(tfrecord, features)

    for i in range(2):
        rec["aug_matrix%s" % i] = tf.io.parse_tensor(
            rec["aug_matrix%s" % i], out_type=tf.float64
        )

    # output a single record containing each augmented example
    record = {}

    # base_features = ["original_token", "idx", "ft_vec"]
    prefixes = ["aug_token", "aug_matrix", "aug_vector", "aug_bool"]
    for p in prefixes:
        tensors = [rec[p + str(i)] for i in range(2)]
        record[p] = tf.stack(tensors)

    for feature in base_features.keys():
        record[feature] = tf.stack([rec[feature]] * 2)

    # encode using binarizer
    reshape_size = (binarizer.word_length * binarizer.encoding_size,)

    aug_token0_encoded = tf.reshape(
        binarizer.binarize(tf.expand_dims(rec["aug_token0"], axis=0)),
        reshape_size,
    )
    aug_token1_encoded = tf.reshape(
        binarizer.binarize(tf.expand_dims(rec["aug_token1"], axis=0)),
        reshape_size,
    )
    original_token_encoded = tf.reshape(
        binarizer.binarize(tf.expand_dims(rec["original_token"], axis=0)),
        reshape_size,
    )

    record["original_encoded"] = tf.stack([original_token_encoded] * 2)
    record["aug_encoded"] = tf.stack([aug_token0_encoded, aug_token1_encoded])
    record["aug_vector"] = record["aug_vector"][:, : binarizer.word_length]
    record["aug_matrix"] = record["aug_matrix"][:, : binarizer.word_length, :]

    flatten = tf.keras.layers.Flatten()
    record["aug_matrix"] = flatten(record["aug_matrix"])
    return record


def Sampler(
    shards_list: List[str],
    binarizer: RETVecBinarizer,
    batch_size: int = 32,
    process_record: Optional[Callable] = None,
    parallelism: int = tf.data.AUTOTUNE,
    file_parallelism: Optional[int] = 1,
    prefetch_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
    compression_type: Optional[str] = None,
) -> tf.data.Dataset:
    """Dataset sampler for RETVec model training.

    Args:
        shards_list: List of input shards.

        binarizer: RetVecBinarizer used to encode words.

        batch_size: Batch size. Defaults to 32.

        process_record: Function to apply to each record after reading.

        parallelism: Number of parallel calls for dataset.map(...).
            Defaults to tf.data.AUTOTUNE.

        file_parallelism: Number of files to read in parallel. Defaults to 1.

        prefetch_size: Num batches to prefetch. Defaults to None.

        buffer_size: Shuffle buffer size. Defaults to None.

        compression_type: Compression type of input shards. Defaults to "GZIP".

    Returns:
        tf.data.Dataset containing the data, processed, shuffle and batched.
    """
    total_shards = len(shards_list)
    print(f"found {total_shards} shards in {time()}.")

    with tf.device("/cpu:0"):
        ds = tf.data.Dataset.from_tensor_slices(shards_list)
        ds = ds.shuffle(total_shards)

        # interleave so we draw examples from different shards
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(
                x, compression_type=compression_type
            ),  # noqa
            block_length=1,  # problem here is that we have non flat record
            num_parallel_calls=file_parallelism,
            cycle_length=file_parallelism,
            deterministic=False,
        )

        # read examples from tfrecord
        ds = ds.map(
            lambda x: read_tfrecord(x, binarizer=binarizer),
            num_parallel_calls=parallelism,
        )

        # ignore corrupted read errors, i.e. corrupted tfrecords
        ds = ds.apply(tf.data.experimental.ignore_errors())

        if buffer_size:
            ds = ds.shuffle(buffer_size)

        ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
        ds = ds.map(process_record, num_parallel_calls=parallelism)
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch_size)
        return ds


def get_process_tfrecord_fn(outputs: Set[str]) -> Callable:
    """Return the transform to process the tfrecord
    and extract only the outputs in `outputs`.
    """

    def process_tfrecord(e):
        x = {"token": e["aug_token"]}
        y = {
            "similarity": e["idx"],
            "ori_decoder": e["original_encoded"],
            "aug_decoder": e["aug_encoded"],
            "aug_vector": e["aug_vector"],
            "aug_matrix": e["aug_matrix"],
        }

        y = {output: y[output] for output in outputs}
        return x, y

    process_tfrecord_fn: Callable = process_tfrecord
    return process_tfrecord_fn


def get_dataset_samplers(
    bucket: str, train_path: str, test_path: str, config: Dict
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Get train and test dataset samplers for training REW* models."""
    core_count = os.cpu_count()

    # Must have at least one CPU
    if not core_count:
        core_count = 1

    client = storage.Client()

    _, _, outputs = get_outputs_info(config)
    tfrecord_fn = get_process_tfrecord_fn(outputs)
    batch_size = config["train"]["batch_size"]
    buffer_size = config["train"]["shuffle_buffer"]
    m = config["model"]
    binarizer = RETVecBinarizer(
        word_length=m["word_length"],
        encoding_size=m["char_encoding_size"],
        encoding_type=m["char_encoding_type"],
        replacement_char=m["replacement_char"],
    )

    train_files = []
    test_files = []

    train_paths = train_path.split(",")
    test_paths = test_path.split(",")

    for path in train_paths:
        for blob in client.list_blobs(bucket, prefix=path):
            if blob.name.endswith(".tfrecord"):
                train_files.append(blob.name)

    for path in test_paths:
        for blob in client.list_blobs(bucket, prefix=path):
            if blob.name.endswith(".tfrecord"):
                test_files.append(blob.name)

    train_shards = []
    test_shards = []

    for f in train_files:
        train_shards.append("gs://" + bucket + "/" + f)

    for f in test_files:
        test_shards.append("gs://" + bucket + "/" + f)

    train_ds = Sampler(
        train_shards,
        binarizer=binarizer,
        process_record=tfrecord_fn,
        batch_size=batch_size,
        file_parallelism=core_count * 2,
        parallelism=core_count,
        buffer_size=buffer_size,
        prefetch_size=1000,
    )

    test_ds = Sampler(
        test_shards,
        binarizer=binarizer,
        process_record=tfrecord_fn,
        file_parallelism=1,
        batch_size=batch_size,
        prefetch_size=1,
    )

    return train_ds, test_ds


def get_outputs_info(
    config: Dict,
) -> Tuple[List[Any], List[List[str]], Set[str]]:
    """Returns the losses, metrics, and output names in the config."""
    loss = []
    metrics: List[List[str]] = []
    outputs = set()

    if config["outputs"].get("similarity_dim"):
        sim_loss_config = config["outputs"]["similarity_loss"]
        sim_loss_type = sim_loss_config["type"]

        if sim_loss_type == "multisim":
            loss.append(
                MultiSimilarityLoss(
                    distance="cosine",
                    alpha=sim_loss_config.get("alpha", 2),
                    beta=sim_loss_config.get("beta", 40),
                    epsilon=sim_loss_config.get("epsilon", 0.1),
                    lmda=sim_loss_config.get("lmda", 0.5),
                )
            )

        elif sim_loss_type == "circle":
            loss.append(
                CircleLoss(
                    distance="cosine",
                    gamma=sim_loss_config.get("gamma", 256),
                    margin=sim_loss_config.get("margin", 0.0),
                )
            )

        elif sim_loss_type == "triplet":
            loss.append(TripletLoss(distance="cosine"))

        elif sim_loss_type == "pn":
            loss.append(PNLoss(distance="cosine"))

        metrics.append([])
        outputs.add("similarity")

    if config["outputs"].get("original_decoder_size"):
        loss.append("binary_crossentropy")
        metrics.append(["mse"])
        outputs.add("ori_decoder")

    if config["outputs"].get("aug_decoder_size"):
        loss.append("binary_crossentropy")
        metrics.append(["mse"])
        outputs.add("aug_decoder")

    if config["outputs"].get("aug_vector_dim"):
        loss.append("binary_crossentropy")
        metrics.append(["acc", "binary_accuracy"])
        outputs.add("aug_vector")

    if config["outputs"].get("aug_matrix_dim"):
        loss.append("binary_crossentropy")
        metrics.append(["acc", "binary_accuracy"])
        outputs.add("aug_matrix")

    return loss, metrics, outputs

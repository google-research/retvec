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

import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_similarity.losses import MultiSimilarityLoss

from retvec import RecVec

# Update this to include code point ranges to be sampled
UNICODE_ALPHABET = [chr(i) for i in range(50000)]


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


def find_primes(max_prime_val):
    """ Find primes between 2 <= p < n """
    if max_prime_val < 6:
        raise ValueError("max_prime_val must be > 6")
    sieve = np.ones(max_prime_val // 3 + (max_prime_val % 6 == 2),
                    dtype=np.bool)
    for i in range(1, int(max_prime_val**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def get_random_unicode(length):
    "generate a random unicode string"
    chars = random.sample(UNICODE_ALPHABET, length)
    return ''.join(chars)


def viz_char_embedding(string,
                       max_len=16,
                       embedding_size=32,
                       positional_encoding=False,
                       cls_int=None):
    """Visualize the CharEmbedding for a given string"

    Args:
        string: string to embedded
        max_len: String max len. Defaults to 16.
        embedding_size: Embedding output size. Defaults to 32.
        positional_encoding: Use positional encoding. Defaults to False.
        cls_int ([type], optional): Add a CLS token to the output.
        Defaults to None.
    """
    ce = RecVec(max_len=max_len,
                is_eager=True,
                embedding_size=embedding_size,
                positional_encoding=positional_encoding,
                cls_int=cls_int)
    embeddings = ce([string])[0]
    plt.imshow(embeddings)


def get_outputs_info(config):
    """Returns the losses, metrics, and output names in the config."""
    loss = []
    metrics = []
    outputs = set()

    if config["outputs"]["similarity_dim"]:
        loss.append(MultiSimilarityLoss('cosine'))
        metrics.append([])
        outputs.add('similarity')

    if config["outputs"]["original_decoder_size"]:
        loss.append('categorical_crossentropy')
        metrics.append(['mse'])
        outputs.add('ori_decoder')

    if config["outputs"]["aug_decoder_size"]:
        loss.append('categorical_crossentropy')
        metrics.append(['mse'])
        outputs.add('aug_decoder')

    if config["outputs"]["complexity_dim"]:
        loss.append('categorical_crossentropy')
        metrics.append(['acc'])
        outputs.add('complexity')

    if config["outputs"]["lang_dim"]:
        loss.append('binary_crossentropy')
        metrics.append(['acc'])
        outputs.add('lang')

    if config["outputs"]["aug_vector_dim"]:
        loss.append('binary_crossentropy')
        metrics.append(['acc'])
        outputs.add('aug_vector')

    if config["outputs"]["aug_matrix_dim"]:
        loss.append('categorical_crossentropy')
        metrics.append(['mse'])
        outputs.add('aug_matrix')

    return loss, metrics, outputs

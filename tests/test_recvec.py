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

from collections import defaultdict

import numpy as np
import pytest
import tensorflow as tf
from retvec import RecVec
from tensorflow.keras import layers

tf.config.set_visible_devices([], 'GPU')


def test_basic_embedding():
    s = "test😀"
    s = tf.constant([s, s])
    print(s)
    for ml in [8, 16, 17, 28]:
        for es in [24, 32, 48]:
            ce = RecVec(max_len=ml, embedding_size=es, is_eager=True)
            embedding = ce(s)
            # batch, max_len, es
            assert embedding.shape == [2, ml, es]
            assert np.max(embedding) == 1
            assert np.min(embedding) == 0


def test_cls_embedding():
    s = "test😀"
    s = tf.constant([s])
    ml = 16
    es = 24
    ce = RecVec(max_len=ml, embedding_size=es, cls_int=3, is_eager=True)
    embedding = ce(s)
    # batch, max_len, es + 1 (cls)
    assert embedding.shape == [1, ml + 1, es]


def test_non_collision():
    # smoke test - if fail something is deeply wrong

    ALPHABET_SIZE = 1024
    chars = [''.join([chr(i) for i in range(ALPHABET_SIZE)])]
    ce = RecVec(max_len=ALPHABET_SIZE, is_eager=True)
    embeddings = ce(chars)

    collisions = defaultdict(int)
    npe = embeddings[0].numpy()
    for idx in range(ALPHABET_SIZE):
        v = "%s" % list(npe[idx])
        collisions[v] += 1

    print(collisions)
    num_collisions = 0
    for v in collisions.values():
        num_collisions += v - 1
    assert not num_collisions


def test_fold_embedding():
    s = "test😀"
    s = tf.constant([s])
    ml = 16
    es = 24
    ce = RecVec(max_len=ml, embedding_size=es, folds=2, is_eager=True)
    embedding = ce(s)
    # batch, max_len, es + 1 (cls)
    assert embedding.shape == [1, ml // 2, es]


def test_odd_fold_embedding():
    s = "test😀"
    s = tf.constant([s])
    ml = 16
    es = 24
    with pytest.raises(ValueError):
        RecVec(max_len=ml, embedding_size=es, folds=3, is_eager=True)


def test_invalid_embedding_size():
    s = "test😀"
    s = tf.constant([s])
    ml = 16
    es = 31  # invalid
    with pytest.raises(ValueError):
        RecVec(max_len=ml, embedding_size=es, folds=3, is_eager=True)


def test_fold_cls_embedding():
    s = "test😀"
    s = tf.constant([s])
    ml = 16
    es = 24
    ce = RecVec(max_len=ml,
                embedding_size=es,
                folds=2,
                cls_int=4,
                is_eager=True)
    embedding = ce(s)
    # batch, max_len, es + 1 (cls)
    assert embedding.shape == [1, ml // 2 + 1, es]


def test_custom_primes_embedding():
    s = "test😀"
    s = tf.constant([s, s])
    primes = [3, 5, 7]
    print(s)
    for ml in [8, 16, 17, 28]:
        for es in [16, 32, 40]:
            ce = RecVec(max_len=ml,
                        embedding_size=es,
                        primes=primes,
                        is_eager=True)
            embedding = ce(s)
            # batch, max_len, es
            assert embedding.shape == [2, ml, es]


def test_determinism():
    s = "test😀"
    s2 = "test😀"
    batch = tf.constant([s, s2])

    ce = RecVec(is_eager=True, lower_case=True)
    embeddings = ce(batch)
    embeddings2 = ce(batch)
    assert tf.reduce_all(tf.equal(embeddings[0], embeddings[1]))
    assert tf.reduce_all(tf.equal(embeddings[0], embeddings2[1]))


def test_lower_case_embedding():
    s = "test😀"
    s = tf.constant([s])

    s2 = "TEST😀"
    s2 = tf.constant([s2])

    ce = RecVec(is_eager=True, lower_case=True)
    embedding = ce(s)
    embedding2 = ce(s2)
    assert tf.reduce_all(tf.equal(embedding, embedding2))


def test_model_building():
    i = layers.Input(shape=(1), dtype='string')
    x = i
    x = RecVec()(x)
    mdl = tf.keras.Model(i, x)
    mdl.summary()
    mdl.predict(['test'])


def test_model_reload(tmp_path):
    i = layers.Input(shape=(1), dtype='string')
    x = i
    x = RecVec(primes=[1, 2, 3, 5, 7])(x)
    mdl = tf.keras.Model(i, x)
    save_path = tmp_path / "char_test_model/"
    mdl.save(save_path)
    tf.keras.models.load_model(save_path)

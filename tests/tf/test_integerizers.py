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

import tensorflow as tf

from retvec.tf.layers import RETVecIntegerizer


def test_graph_mode():
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecIntegerizer(max_chars=16)(i)
    model = tf.keras.models.Model(i, x)

    test_inputs = [
        tf.constant(["Testing😀"]),
        tf.constant(["Testing😀", "Testing😀"]),
        tf.constant(["Testing a very long string as input"]),
    ]

    for test_input in test_inputs:
        embeddings = model(test_input)
        assert embeddings.shape == (test_input.shape[0], 16)


def test_eager_mode():
    intergerizer = RETVecIntegerizer(max_chars=16)

    embeddings = intergerizer.integerize(tf.constant("Testing😀"))
    assert embeddings.shape == [16]

    embeddings = intergerizer.integerize(tf.constant(["Testing😀", "Testing😀"]))
    assert embeddings.shape == [2, 16]


def test_2d_inputs():
    i = tf.keras.layers.Input((2,), dtype=tf.string)
    x = RETVecIntegerizer(max_chars=16)(i)
    model = tf.keras.models.Model(i, x)

    test_input = tf.constant([["a", "b"], ["c", "d"]])

    embeddings = model(test_input)
    assert embeddings.shape == (2, 2, 16)


def test_tfds_map():
    intergerizer = RETVecIntegerizer(max_chars=16)

    dataset = tf.data.Dataset.from_tensor_slices(["Testing😀", "Testing😀"])
    dataset = dataset.map(intergerizer.integerize)

    for ex in dataset.take(1):
        assert ex.shape == [16]

    dataset = tf.data.Dataset.from_tensor_slices(["Testing😀", "Testing😀"])
    dataset = dataset.repeat()
    dataset = dataset.batch(2)
    dataset = dataset.map(intergerizer.integerize)

    for ex in dataset.take(1):
        assert ex.shape == [2, 16]


def test_determinism_eager_mode():
    intergerizer = RETVecIntegerizer(max_chars=16)

    s = "Testing😀"
    test_input = tf.constant([s, s])

    embeddings = intergerizer.integerize(test_input)
    embeddings2 = intergerizer.integerize(test_input)

    assert tf.reduce_all(tf.equal(embeddings[0], embeddings[1]))
    assert tf.reduce_all(tf.equal(embeddings[0], embeddings2[1]))


def test_determinism_graph_mode():
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecIntegerizer(max_chars=16)(i)
    model = tf.keras.models.Model(i, x)

    s = "Testing😀"
    test_input = tf.constant([s, s])

    embeddings = model(test_input)
    embeddings2 = model(test_input)

    assert tf.reduce_all(tf.equal(embeddings[0], embeddings[1]))
    assert tf.reduce_all(tf.equal(embeddings[0], embeddings2[1]))


def test_serialization(tmp_path):
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecIntegerizer(max_chars=16)(i)
    model = tf.keras.models.Model(i, x)

    save_path = tmp_path / "test_serialization_integerizer"
    model.save(save_path)
    tf.keras.models.load_model(save_path)


def test_common_parameters():
    test_input = tf.constant(["Testing😀", "Testing😀"])

    for max_chars in [8, 16, 32]:
        for encoding_type in ["UTF-8", "UTF-16-BE"]:
            for replacement_int in [0, 65533]:
                i = tf.keras.layers.Input((1,), dtype=tf.string)
                x = RETVecIntegerizer(
                    max_chars=max_chars,
                    encoding_type=encoding_type,
                    replacement_int=replacement_int,
                )(i)
                model = tf.keras.models.Model(i, x)

                embedding = model(test_input)
                assert embedding.shape == (2, max_chars)

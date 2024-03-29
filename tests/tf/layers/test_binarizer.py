"""
 Copyright 2023 Google LLC

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

import pytest
import tensorflow as tf

from retvec.tf.layers import RETVecBinarizer, RETVecIntToBinary

use_native = [True, False]
use_native_names = ["native_tf", "tf"]


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_graph_mode(use_tf_lite_compatible_ops):
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecBinarizer(
        word_length=16,
        encoding_size=24,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )(i)
    model = tf.keras.models.Model(i, x)

    test_inputs = [
        tf.constant(["Testing😀"]),
        tf.constant(["Testing😀", "Testing😀"]),
        tf.constant(["Testing a very long string as input"]),
    ]

    for test_input in test_inputs:
        embeddings = model(test_input)
        assert embeddings.shape == (test_input.shape[0], 16, 24)


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_eager_mode(use_tf_lite_compatible_ops):
    binarizer = RETVecBinarizer(
        word_length=16,
        encoding_size=24,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )

    s = "Testing😀"

    embeddings = binarizer.binarize(tf.constant(s))
    assert embeddings.shape == [16, 24]

    embeddings = binarizer.binarize(tf.constant([s, s, s]))
    assert embeddings.shape == [3, 16, 24]


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_2d_inputs(use_tf_lite_compatible_ops):
    i = tf.keras.layers.Input((2,), dtype=tf.string)
    x = RETVecBinarizer(
        word_length=16,
        encoding_size=24,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )(i)
    model = tf.keras.models.Model(i, x)

    test_input = tf.constant([["a", "b"], ["c", "d"]])

    embeddings = model(test_input)
    assert embeddings.shape == (2, 2, 16, 24)


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_tfds_map(use_tf_lite_compatible_ops):
    binarizer = RETVecBinarizer(
        word_length=16,
        encoding_size=24,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )

    dataset = tf.data.Dataset.from_tensor_slices(["Testing😀", "Testing😀"])
    dataset = dataset.map(binarizer.binarize)

    for ex in dataset.take(1):
        assert ex.shape == [16, 24]

    dataset = tf.data.Dataset.from_tensor_slices(["Testing😀", "Testing😀"])
    dataset = dataset.repeat()
    dataset = dataset.batch(2)
    dataset = dataset.map(binarizer.binarize)

    for ex in dataset.take(1):
        assert ex.shape == [2, 16, 24]


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_determinism_eager_mode(use_tf_lite_compatible_ops):
    binarizer = RETVecBinarizer(
        word_length=16,
        encoding_size=24,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )

    s = "Testing😀"
    test_input = tf.constant([s, s])

    embeddings = binarizer.binarize(test_input)
    embeddings2 = binarizer.binarize(test_input)

    assert tf.reduce_all(tf.equal(embeddings[0], embeddings[1]))
    assert tf.reduce_all(tf.equal(embeddings[0], embeddings2[1]))


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_determinism_graph_mode(use_tf_lite_compatible_ops):
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecBinarizer(
        word_length=16,
        encoding_size=24,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )(i)
    model = tf.keras.models.Model(i, x)

    s = "Testing😀"
    test_input = tf.constant([s, s])

    embeddings = model(test_input)
    embeddings2 = model(test_input)

    assert tf.reduce_all(tf.equal(embeddings[0], embeddings[1]))
    assert tf.reduce_all(tf.equal(embeddings[0], embeddings2[1]))


def test_serialization(tmp_path):
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecBinarizer(word_length=16, encoding_size=24)(i)
    model = tf.keras.models.Model(i, x)

    save_path = tmp_path / "test_serialization_binarizer"
    model.save(save_path)
    tf.keras.models.load_model(save_path)


def test_native_values():
    test_input = tf.constant(["Testing😀", "Testing😀"])

    for word_length in [8, 16, 32]:
        for encoding_size in [16, 24]:
            for encoding_type in ["UTF-8", "UTF-16-BE"]:
                for replacement_char in [0, 65533]:
                    i = tf.keras.layers.Input((1,), dtype=tf.string)
                    x = RETVecBinarizer(
                        word_length=word_length,
                        encoding_size=encoding_size,
                        encoding_type=encoding_type,
                        replacement_char=replacement_char,
                        use_tf_lite_compatible_ops=False,
                    )(i)
                    model = tf.keras.models.Model(i, x)

                    embedding = model(test_input)
                    assert embedding.shape == (2, word_length, encoding_size)

                    x = RETVecBinarizer(
                        word_length=word_length,
                        encoding_size=encoding_size,
                        encoding_type=encoding_type,
                        replacement_char=replacement_char,
                        use_tf_lite_compatible_ops=True,
                    )(i)
                    model = tf.keras.models.Model(i, x)
                    embedding_native = model(test_input)

                    assert embedding_native.shape == embedding.shape
                    assert (
                        embedding.numpy().tolist()
                        == embedding_native.numpy().tolist()
                    )


def test_encoding_values():
    i = tf.keras.layers.Input((8,), dtype=tf.int32)
    x = RETVecIntToBinary(sequence_length=1, word_length=8, encoding_size=24)(
        i
    )
    model = tf.keras.models.Model(i, x)

    test_inputs = tf.constant(
        [[100, 65536, 65535, 1114111, 2000000, 2**24 - 1, -1, 2**24]]
    )

    binary_encodings = model(test_inputs)

    expected_output = tf.constant(
        [
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ],
        dtype=tf.float32,
    )

    assert tf.reduce_all(tf.equal(expected_output, binary_encodings))

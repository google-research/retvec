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

from retvec.tf.layers import RETVecBinarizer, RETVecEmbedding

TEST_EMB_SIZE = 256
TEST_WORD_LENGTH = 16
TEST_CHAR_ENCODING_SIZE = 24
TEST_INPUTS = [
    tf.constant(["Testing😀"]),
    tf.constant(["Testing😀", "Testing😀"]),
    tf.constant(["Testing a very long string as input"]),
]


def create_retvec_embedding(tmp_path):
    i = tf.keras.layers.Input(
        (TEST_WORD_LENGTH, TEST_CHAR_ENCODING_SIZE), dtype=tf.float32
    )
    x = tf.keras.layers.Flatten()(i)
    o = tf.keras.layers.Dense(TEST_EMB_SIZE)(x)
    model = tf.keras.models.Model(i, o)

    save_path = tmp_path / "test_retvec_embedding"
    model.save(save_path)

    embedding_model = RETVecEmbedding(str(save_path))
    return embedding_model


def test_rewnet_model(tmp_path):
    embedding_model = create_retvec_embedding(tmp_path)
    binarizer = RETVecBinarizer(
        word_length=TEST_WORD_LENGTH, encoding_size=TEST_CHAR_ENCODING_SIZE
    )

    for test_input in TEST_INPUTS:
        embeddings = embedding_model(binarizer.binarize(test_input))
        assert embeddings.shape == (test_input.shape[0], TEST_EMB_SIZE)


def test_2d_inputs(tmp_path):
    embedding_model = create_retvec_embedding(tmp_path)

    test_input = tf.random.uniform(
        (2, 3, TEST_WORD_LENGTH, TEST_CHAR_ENCODING_SIZE),
        minval=0,
        maxval=2,
        dtype=tf.int32,
    )
    test_input = tf.cast(test_input, dtype=tf.float32)
    embeddings = embedding_model(test_input)
    assert embeddings.shape == (test_input.shape[0], 3, TEST_EMB_SIZE)


def test_binarizer_embedding_model(tmp_path):
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecBinarizer(
        word_length=TEST_WORD_LENGTH, encoding_size=TEST_CHAR_ENCODING_SIZE
    )(i)
    o = create_retvec_embedding(tmp_path)(x)
    model = tf.keras.models.Model(i, o)

    for test_input in TEST_INPUTS:
        embeddings = model(test_input)
        assert embeddings.shape == (test_input.shape[0], TEST_EMB_SIZE)


def test_binarizer_embedding_model_2d(tmp_path):
    i = tf.keras.layers.Input((3,), dtype=tf.string)
    x = RETVecBinarizer(
        word_length=TEST_WORD_LENGTH, encoding_size=TEST_CHAR_ENCODING_SIZE
    )(i)
    o = create_retvec_embedding(tmp_path)(x)
    model = tf.keras.models.Model(i, o)

    test_input = tf.constant([["a", "b", "c"], ["d", "e", "f"]])

    embeddings = model(test_input)
    assert embeddings.shape == (
        test_input.shape[0],
        test_input.shape[1],
        TEST_EMB_SIZE,
    )


def test_serialization(tmp_path):
    embedding_model = create_retvec_embedding(tmp_path)

    i = tf.keras.layers.Input(
        (TEST_WORD_LENGTH, TEST_CHAR_ENCODING_SIZE), dtype=tf.float32
    )
    x = embedding_model(i)
    model = tf.keras.models.Model(i, x)

    save_path = tmp_path / "test_retvec_embedding_serialization"
    model.save(save_path)
    tf.keras.models.load_model(save_path)


def test_default_embedding_model(tmp_path):
    embedding_size = 256
    binarizer = RETVecBinarizer(
        word_length=TEST_WORD_LENGTH, encoding_size=TEST_CHAR_ENCODING_SIZE
    )

    i = tf.keras.layers.Input(
        (TEST_WORD_LENGTH, TEST_CHAR_ENCODING_SIZE), dtype=tf.float32
    )
    x = RETVecEmbedding(model="retvec-v1")(i)
    model = tf.keras.models.Model(i, x)

    for test_input in TEST_INPUTS:
        embeddings = model(binarizer.binarize(test_input))
        assert embeddings.shape == (test_input.shape[0], embedding_size)

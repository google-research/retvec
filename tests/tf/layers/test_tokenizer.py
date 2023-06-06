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

from retvec.tf.layers import RETVecTokenizer

SEQUENCE_LENGTH = 128
WORD_LENGTH = 16
CHAR_ENCODING_SIZE = 24
RETVEC_MODEL = "retvec-v1"


def test_graph_mode_with_model(tmp_path):
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecTokenizer(
        sequence_length=SEQUENCE_LENGTH,
        model=RETVEC_MODEL,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )(i)
    model = tf.keras.models.Model(i, x)

    test_inputs = [
        tf.constant(["Testing😀 a full sentence"]),
        tf.constant(["Testing😀", "Testing😀"]),
    ]

    for test_input in test_inputs:
        embeddings = model(test_input)
        assert embeddings.shape == (
            test_input.shape[0],
            SEQUENCE_LENGTH,
            256,
        )


def test_eager_mode_with_model(tmp_path):
    tokenizer = RETVecTokenizer(
        model=RETVEC_MODEL,
        sequence_length=SEQUENCE_LENGTH,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )

    s = "Testing😀 a full sentence"

    embeddings = tokenizer.tokenize(tf.constant(s))
    assert embeddings.shape == [SEQUENCE_LENGTH, tokenizer.embedding_size]

    embeddings = tokenizer.tokenize(tf.constant([s, s, s]))
    assert embeddings.shape == [3, SEQUENCE_LENGTH, tokenizer.embedding_size]


def test_graph_mode_no_model():
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RETVecTokenizer(
        model=None,
        sequence_length=SEQUENCE_LENGTH,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )(i)
    model = tf.keras.models.Model(i, x)

    test_inputs = [
        tf.constant(["Testing😀 a full sentence"]),
        tf.constant(["Testing😀", "Testing😀"]),
    ]

    for test_input in test_inputs:
        embeddings = model(test_input)
        assert embeddings.shape == (
            test_input.shape[0],
            SEQUENCE_LENGTH,
            WORD_LENGTH * CHAR_ENCODING_SIZE,
        )


def test_eager_mode_no_model():
    tokenizer = RETVecTokenizer(
        model=None,
        sequence_length=SEQUENCE_LENGTH,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )

    assert tokenizer.embedding_size == WORD_LENGTH * CHAR_ENCODING_SIZE
    s = "Testing😀 a full sentence"

    embeddings = tokenizer.tokenize(tf.constant(s))
    assert embeddings.shape == [SEQUENCE_LENGTH, tokenizer.embedding_size]

    embeddings = tokenizer.tokenize(tf.constant([s, s, s]))
    assert embeddings.shape == [3, SEQUENCE_LENGTH, tokenizer.embedding_size]


def test_standardize():
    for standardize in [
        None,
        "lower_and_strip_punctuation",
        "strip_punctuation",
        "lower",
    ]:
        tokenizer = RETVecTokenizer(
            model=None,
            sequence_length=SEQUENCE_LENGTH,
            word_length=WORD_LENGTH,
            char_encoding_size=CHAR_ENCODING_SIZE,
            standardize=standardize,
        )
        s = "Testing 😀 a full sentence!"

        embeddings = tokenizer.tokenize(tf.constant(s))
        assert embeddings.shape == [SEQUENCE_LENGTH, tokenizer.embedding_size]


def test_tfds_map_tokenize(tmp_path):
    for model_path in [None, RETVEC_MODEL]:
        tokenizer = RETVecTokenizer(
            model=model_path,
            sequence_length=SEQUENCE_LENGTH,
            word_length=WORD_LENGTH,
            char_encoding_size=CHAR_ENCODING_SIZE,
        )

        dataset = tf.data.Dataset.from_tensor_slices(["Testing😀"])
        dataset = dataset.map(tokenizer.tokenize)

        for ex in dataset.take(1):
            assert ex.shape == [SEQUENCE_LENGTH, tokenizer.embedding_size]

        dataset = tf.data.Dataset.from_tensor_slices(["Testing😀"])
        dataset = dataset.repeat()
        dataset = dataset.batch(2)
        dataset = dataset.map(tokenizer.tokenize)

        for ex in dataset.take(1):
            assert ex.shape == [2, SEQUENCE_LENGTH, tokenizer.embedding_size]


def test_serialization(tmp_path):
    for model_path in [None, RETVEC_MODEL]:
        i = tf.keras.layers.Input((1,), dtype=tf.string)
        x = RETVecTokenizer(
            model=model_path,
            sequence_length=SEQUENCE_LENGTH,
            word_length=WORD_LENGTH,
            char_encoding_size=CHAR_ENCODING_SIZE,
        )(i)
        model = tf.keras.models.Model(i, x)

        save_path = tmp_path / "test_retvec_serialization"
        model.save(save_path)
        tf.keras.models.load_model(save_path)

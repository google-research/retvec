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

from tensorflow_retvec import RetVec

MAX_LEN = 128
MAX_CHARS = 16
CHAR_ENCODING_SIZE = 32
EMBEDDING_SIZE = 128


def create_and_save_retvec_embedding(tmp_path):
    i = tf.keras.layers.Input((MAX_CHARS, CHAR_ENCODING_SIZE), dtype=tf.float32)
    x = tf.keras.layers.Flatten()(i)
    o = tf.keras.layers.Dense(EMBEDDING_SIZE)(x)
    model = tf.keras.models.Model(i, o)

    save_path = tmp_path / "test_retvec_embedding"
    model.save(save_path)
    return str(save_path)


def test_graph_mode_with_model(tmp_path):
    model_path = create_and_save_retvec_embedding(tmp_path)

    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RetVec(
        model=model_path,
        max_len=MAX_LEN,
        max_chars=MAX_CHARS,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )(i)
    model = tf.keras.models.Model(i, x)

    test_inputs = [
        tf.constant(["Testing😀 a full sentence"]),
        tf.constant(["Testing😀", "Testing😀"]),
    ]

    for test_input in test_inputs:
        embeddings = model(test_input)
        assert embeddings.shape == (test_input.shape[0], MAX_LEN, EMBEDDING_SIZE)


def test_eager_mode_with_model(tmp_path):
    model_path = create_and_save_retvec_embedding(tmp_path)

    tokenizer = RetVec(
        model=model_path,
        max_len=MAX_LEN,
        max_chars=MAX_CHARS,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )
    assert tokenizer.embedding_size == EMBEDDING_SIZE

    s = "Testing😀 a full sentence"

    embeddings = tokenizer.tokenize(tf.constant(s))
    assert embeddings.shape == [MAX_LEN, EMBEDDING_SIZE]

    embeddings = tokenizer.tokenize(tf.constant([s, s, s]))
    assert embeddings.shape == [3, MAX_LEN, EMBEDDING_SIZE]


def test_graph_mode_no_model():
    i = tf.keras.layers.Input((1,), dtype=tf.string)
    x = RetVec(
        model=None,
        max_len=MAX_LEN,
        max_chars=MAX_CHARS,
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
            MAX_LEN,
            MAX_CHARS * CHAR_ENCODING_SIZE,
        )


def test_eager_mode_no_model():
    tokenizer = RetVec(
        model=None,
        max_len=MAX_LEN,
        max_chars=MAX_CHARS,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )

    assert tokenizer.embedding_size == MAX_CHARS * CHAR_ENCODING_SIZE
    s = "Testing😀 a full sentence"

    embeddings = tokenizer.tokenize(tf.constant(s))
    assert embeddings.shape == [MAX_LEN, tokenizer.embedding_size]

    embeddings = tokenizer.tokenize(tf.constant([s, s, s]))
    assert embeddings.shape == [3, MAX_LEN, tokenizer.embedding_size]


def test_tfds_map_tokenize(tmp_path):
    model_path = create_and_save_retvec_embedding(tmp_path)

    for model in [None, model_path]:
        tokenizer = RetVec(
            model=model,
            max_len=MAX_LEN,
            max_chars=MAX_CHARS,
            char_encoding_size=CHAR_ENCODING_SIZE,
        )

        dataset = tf.data.Dataset.from_tensor_slices(["Testing😀"])
        dataset = dataset.map(tokenizer.tokenize)

        for ex in dataset.take(1):
            assert ex.shape == [MAX_LEN, tokenizer.embedding_size]

        dataset = tf.data.Dataset.from_tensor_slices(["Testing😀"])
        dataset = dataset.repeat()
        dataset = dataset.batch(2)
        dataset = dataset.map(tokenizer.tokenize)

        for ex in dataset.take(1):
            assert ex.shape == [2, MAX_LEN, tokenizer.embedding_size]


def test_tfds_map_binarize(tmp_path):
    tokenizer = RetVec(
        model=None,
        max_len=MAX_LEN,
        max_chars=MAX_CHARS,
        char_encoding_size=CHAR_ENCODING_SIZE,
    )

    dataset = tf.data.Dataset.from_tensor_slices(["Testing😀", "Testing😀"])
    dataset = dataset.map(tokenizer.binarize)

    for ex in dataset.take(1):
        assert ex.shape == [MAX_CHARS, CHAR_ENCODING_SIZE]

    dataset = tf.data.Dataset.from_tensor_slices(["Testing😀", "Testing😀"])
    dataset = dataset.repeat()
    dataset = dataset.batch(2)
    dataset = dataset.map(tokenizer.binarize)

    for ex in dataset.take(1):
        assert ex.shape == [2, MAX_CHARS, CHAR_ENCODING_SIZE]


def test_serialization(tmp_path):
    model_path = create_and_save_retvec_embedding(tmp_path)

    for model in [None, model_path]:
        i = tf.keras.layers.Input((1,), dtype=tf.string)
        x = RetVec(
            model=model,
            max_len=MAX_LEN,
            max_chars=MAX_CHARS,
            char_encoding_size=CHAR_ENCODING_SIZE,
        )(i)
        model = tf.keras.models.Model(i, x)

        save_path = tmp_path / "test_retvec_serialization"
        model.save(save_path)
        tf.keras.models.load_model(save_path)

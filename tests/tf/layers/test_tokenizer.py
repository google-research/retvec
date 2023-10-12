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

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.lite.python import interpreter

from retvec.tf.layers import RETVecTokenizer

use_native = [True, False]
use_native_names = ["native_tf", "tf"]

SEQUENCE_LENGTH = 128
WORD_LENGTH = 16
CHAR_ENCODING_SIZE = 24
RETVEC_MODEL = "retvec-v1"


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_graph_mode_with_model(use_tf_lite_compatible_ops):
    i = tf.keras.Input((1,), dtype=tf.string)
    x = RETVecTokenizer(
        sequence_length=SEQUENCE_LENGTH,
        model=RETVEC_MODEL,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
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


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_eager_mode_with_model(use_tf_lite_compatible_ops):
    tokenizer = RETVecTokenizer(
        model=RETVEC_MODEL,
        sequence_length=SEQUENCE_LENGTH,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
    )

    s = "Testing😀 a full sentence"

    embeddings = tokenizer.tokenize(tf.constant(s))
    assert embeddings.shape == [SEQUENCE_LENGTH, tokenizer.embedding_size]

    embeddings = tokenizer.tokenize(tf.constant([s, s, s]))
    assert embeddings.shape == [3, SEQUENCE_LENGTH, tokenizer.embedding_size]


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_graph_mode_no_model(use_tf_lite_compatible_ops):
    i = tf.keras.Input((1,), dtype=tf.string)
    x = RETVecTokenizer(
        model=None,
        sequence_length=SEQUENCE_LENGTH,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
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


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_eager_mode_no_model(use_tf_lite_compatible_ops):
    tokenizer = RETVecTokenizer(
        model=None,
        sequence_length=SEQUENCE_LENGTH,
        word_length=WORD_LENGTH,
        char_encoding_size=CHAR_ENCODING_SIZE,
        use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
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


@pytest.mark.parametrize(
    "use_tf_lite_compatible_ops", use_native, ids=use_native_names
)
def test_tfds_map_tokenize(use_tf_lite_compatible_ops):
    for model_path in [None, RETVEC_MODEL]:
        tokenizer = RETVecTokenizer(
            model=model_path,
            sequence_length=SEQUENCE_LENGTH,
            word_length=WORD_LENGTH,
            char_encoding_size=CHAR_ENCODING_SIZE,
            use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
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


def test_tf_lite_conversion():
    for model_path in [None, RETVEC_MODEL]:
        i = tf.keras.layers.Input((1,), dtype=tf.string, name="input")
        x = RETVecTokenizer(
            model=model_path,
            sequence_length=SEQUENCE_LENGTH,
            word_length=WORD_LENGTH,
            char_encoding_size=CHAR_ENCODING_SIZE,
            use_tf_lite_compatible_ops=True,
        )(i)
        model = tf.keras.models.Model(i, {"tokens": x})

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.allow_custom_ops = True
        tflite_model = converter.convert()

        # Perform TensorFlow Lite inference.
        interp = interpreter.InterpreterWithCustomOps(
            model_content=tflite_model,
            custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS,
        )
        interp.get_signature_list()

        input_data = np.array(
            ["Some minds are better kept apart", "this is a test"]
        )

        tokenize = interp.get_signature_runner("serving_default")
        output = tokenize(input=input_data)

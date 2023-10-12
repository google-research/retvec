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

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers

try:
    from tensorflow_text import WhitespaceTokenizer, utf8_binarize
except ImportError:
    WhitespaceTokenizer = None
    utf8_binarize = None

from .binarizer import RETVecBinarizer, _reshape_embeddings
from .embedding import RETVecEmbedding

LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"
STRIP_PUNCTUATION = "strip_punctuation"
LOWER = "lower"


# This is an explicit regex of all the tokens that will be stripped if
# LOWER_AND_STRIP_PUNCTUATION is set. If an application requires other
# stripping, a Callable should be passed into the 'standardize' arg.
DEFAULT_STRIP_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'


@tf.keras.utils.register_keras_serializable(package="retvec")
class RETVecTokenizer(tf.keras.layers.Layer):
    """RETVec: Resilient and Efficient Text Vectorizer layer

    This layer is typically placed as the first layer after the
    input layer in a model, to convert raw input text into
    sequences of RETVec embeddings. This layer is very efficient on
    GPU and CPU. For running RETVec most efficiently on TPU,
    please see the RETVec on TPU tutorial notebook.

    Example usage:
        i = tf.keras.layers.Input((1,), dtype=tf.string)
        x = RETVecTokenizer(sequence_length=512, model=optional_model_path)(i)
        ...
        [Build the remainder of the model, i.e. BERT or LSTM encoder]
        ...

    """

    def __init__(
        self,
        sequence_length: int = 128,
        model: Optional[Union[str, Path]] = "retvec-v1",
        trainable: bool = False,
        sep: str = "",
        standardize: Optional[str] = None,
        use_tf_lite_compatible_ops: bool = False,
        word_length: int = 16,
        char_encoding_size: int = 24,
        char_encoding_type: str = "UTF-8",
        replacement_char: int = 65533,
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        norm_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize a RETVec layer.

        Args:
            sequence_length: Maximum number of words per sequence. If the input
                text is longer than `sequence_length` words after split by
                `sep` separator, the input will be truncated to
                `sequence_length` words.

            model: Path to saved pretrained RETVec model, str or pathlib.Path
                object. "retvec-v1" to use V1 of the pre-trained RETVec word
                embedding model, None to use the default RETVec character
                encoding.

            trainable: Whether to make the pretrained RETVec model trainable
                or to freeze all weights.

            sep: Separator to split input text on into words. Defaults to '',
                which splits on all whitespace and stripts whitespace from
                beginning or end of the text. See tf.strings.split for more
                details.

            standardize: Optional specification for standardization to apply to
                the input text. One of None, "lower_and_strip_punctuation",
                "lower", "strip_punctuation", or a callable function which
                applies standardization and returns a tf.string Tensor.

            use_tf_lite_compatible_ops: A boolean indicating whether to only
                TF Lite compatible ops that are supported natively in TF.
                `sep` and `standardize` will not be used and whitespace
                splitting will be always used, so preprocessing such
                as lowercasing should happen before the text is passed
                into this layer.

            word_length: The number of characters per word to embed.
                The integer representation of each word will be padded or
                truncated to be `word_length` characters.

                Note: if you are using a pretrained RETVec model,
                `word_length` must match the word length used in
                the model or else it will break.

            char_encoding_size: Number of floats used to encode each
                character in the binary representation. Defaults to 24,
                which ensures that all UTF-8 codepoints can be uniquely
                represented.

                Note: if you are using a pretrained RETVec model,
                `encoding_size` must match the encoding size used in
                the model or else it will break.

            char_encoding_type: String name for the unicode encoding that
                should be used to decode each string.

            replacement_char: The replacement Unicode integer codepoint to be
                used in place of invalid substrings in the input.

            dropout_rate: Dropout rate to apply on RETVec embedding.

            spatial_dropout_rate: Spatial dropout rate to apply on RETVec
                embedding.

            norm_type: Norm to apply on RETVec embedding. One of
                [None, 'batch', or 'layer'].

            **kwargs: Additional keyword args passed to the base Layer class.
        """
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.sep = sep
        self.standardize = standardize
        self.use_tf_lite_compatible_ops = use_tf_lite_compatible_ops
        self.model = model
        self.trainable = trainable

        # Use whitesapce tokenizer for TF Lite compatibility
        # TODO (marinazh): use TF Text functions like regex_split to offer
        # more flexibility and preprocessing options
        self._native_mode = (
            self.use_tf_lite_compatible_ops
            and WhitespaceTokenizer
            and utf8_binarize
        )

        if use_tf_lite_compatible_ops and not self._native_mode:
            logging.warning(
                "Native support for `RETVecTokenizer` unavailable. "
                "Check `tensorflow_text.utf8_binarize` availability"
                " and its parameter contraints."
            )

        if self._native_mode:
            self._whitespace_tokenizer = WhitespaceTokenizer()

        # RetVecEmbedding
        if self.model:
            self._embedding: Optional[RETVecEmbedding] = RETVecEmbedding(
                model=model, trainable=self.trainable
            )
        else:
            self._embedding = None

        # RetVecBinarizer
        self.word_length = word_length
        self.char_encoding_size = char_encoding_size
        self.char_encoding_type = char_encoding_type
        self.replacement_char = replacement_char
        self._binarizer = RETVecBinarizer(
            word_length=self.word_length,
            encoding_size=self.char_encoding_size,
            encoding_type=self.char_encoding_type,
            replacement_char=self.replacement_char,
            use_tf_lite_compatible_ops=use_tf_lite_compatible_ops,
        )

        # Set to True when 'tokenize()' or 'binarize()' called in eager mode
        self.eager = False

        if self._embedding:
            self._embedding_size = self._embedding.embedding_size
        else:
            self._embedding = None
            self._embedding_size = self.word_length * self.char_encoding_size

        # Create post-embedding layers
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.norm_type = norm_type

        if self.norm_type == "batch":
            self.norm = layers.BatchNormalization()
        elif self.norm_type == "layer":
            self.norm = layers.LayerNormalization()
        elif self.norm_type:
            raise ValueError(f"Unsupported norm_type {self.norm_type}")

        self.dropout = layers.Dropout(self.dropout_rate)
        self.spatial_drop = layers.SpatialDropout1D(self.spatial_dropout_rate)

    @property
    def embedding(self):
        return self._embedding

    @property
    def binarizer(self):
        return self._binarizer

    @property
    def embedding_size(self):
        return self._embedding_size

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:
        inputs = tf.stop_gradient(inputs)
        batch_size = tf.shape(inputs)[0]

        if self._native_mode:
            # ensure batch of tf.strings doesn't have extra dim
            if len(inputs.shape) == 2:
                inputs = tf.squeeze(inputs, axis=1)

            # whitespace tokenization
            tokenized = self._whitespace_tokenizer.tokenize(inputs)
            row_lengths = tokenized.row_lengths()

            # apply native binarization op
            # NOTE: utf8_binarize used here because RaggedTensorToTensor isn't
            # supported in TF Lite for strings, this is a workaround
            binarized = utf8_binarize(tokenized.flat_values)
            binarized = tf.RaggedTensor.from_row_lengths(
                values=binarized, row_lengths=row_lengths
            )

            # convert from RaggedTensor to Tensor
            binarized = binarized.to_tensor(
                default_value=0,
                shape=(
                    batch_size,
                    self.sequence_length,
                    self.word_length * self.char_encoding_size,
                ),
            )

            # reshape embeddings to apply the RETVecEmbedding layer
            binarized = _reshape_embeddings(
                binarized,
                batch_size=batch_size,
                sequence_length=self.sequence_length,
                word_length=self.word_length,
                encoding_size=self.char_encoding_size,
            )

        else:
            # standardize and preprocess text
            if self.standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
                inputs = tf.strings.lower(inputs)
            if self.standardize in (
                STRIP_PUNCTUATION,
                LOWER_AND_STRIP_PUNCTUATION,
            ):
                inputs = tf.strings.regex_replace(
                    inputs, DEFAULT_STRIP_REGEX, ""
                )
            if callable(self.standardize):
                inputs = self.standardize(inputs)

            # split text on separator
            rtensor = tf.strings.split(
                inputs, sep=self.sep, maxsplit=self.sequence_length
            )

            # Handle shape differences between eager and graph mode
            if self.eager:
                stensor = rtensor.to_tensor(
                    default_value="",
                    shape=(rtensor.shape[0], self.sequence_length),
                )
            else:
                stensor = rtensor.to_tensor(
                    default_value="",
                    shape=(rtensor.shape[0], 1, self.sequence_length),
                )
                stensor = tf.squeeze(stensor, axis=1)

            # apply RETVec binarization
            binarized = self._binarizer(stensor, training=training)

        # embed using RETVec word embedding model, if available
        if self._embedding:
            embeddings = self._embedding(binarized, training=training)
        else:
            embsize = self.char_encoding_size * self.word_length
            embeddings = tf.reshape(
                binarized, (tf.shape(inputs)[0], self.sequence_length, embsize)
            )

        # apply post-embedding norm and dropout layers
        if self.norm_type:
            embeddings = self.norm(embeddings, training=training)

        if self.dropout_rate:
            embeddings = self.dropout(embeddings, training=training)

        if self.spatial_dropout_rate:
            embeddings = self.spatial_drop(embeddings, training=training)

        return embeddings

    def tokenize(self, inputs: Tensor) -> Tensor:
        """Return RetVec embeddings for a word or a list of words.

        Args:
            inputs: A single word or list of words to encode.

        Returns:
            Retvec embeddings for the input words(s).
        """
        inputs_shape = inputs.shape
        if inputs_shape == tf.TensorShape([]):
            inputs = tf.expand_dims(inputs, 0)

        # set layers to eager mode
        self.eager = True
        self._binarizer.eager = True

        # compute embeddings
        embeddings = self(inputs, training=False)

        # Remove extra dim if input was a single word
        if inputs_shape == tf.TensorShape([]):
            embeddings = tf.squeeze(embeddings)

        return embeddings

    def detokenize(self, inputs: Tensor):
        raise NotImplementedError()

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "sep": self.sep,
                "standardize": self.standardize,
                "use_tf_lite_compatible_ops": self.use_tf_lite_compatible_ops,
                "model": self.model,
                "trainable": self.trainable,
                "word_length": self.word_length,
                "char_encoding_size": self.char_encoding_size,
                "char_encoding_type": self.char_encoding_type,
                "replacement_char": self.replacement_char,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "norm_type": self.norm_type,
            }
        )
        return config

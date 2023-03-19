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

import logging
import re
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow import Tensor, TensorShape

try:
    from tensorflow_text import utf8_binarize
except ImportError:
    utf8_binarize = None

from .integerizers import RetVecIntegerizer


def _reshape_embeddings(embeddings: tf.Tensor, batch_size: int,
                        sequence_length: int, word_length: int,
                        encoding_size: int) -> tf.Tensor:
    if sequence_length > 1:
        return tf.reshape(
            embeddings,
            (
                batch_size,
                sequence_length,
                word_length,
                encoding_size,
            ),
        )
    else:
        return tf.reshape(embeddings,
                          (batch_size, word_length, encoding_size))


@tf.keras.utils.register_keras_serializable(package="retvec")
class RetVecIntToBinary(tf.keras.layers.Layer):
    """Convert UTF-8 code points tensor into their float binary representation.
    """

    def __init__(self, sequence_length: int = 1, word_length: int = 16,
                 encoding_size: int = 32, **kwargs) -> None:
        """Initialize a RetVec integer binarizer.

        Args:
            #FIXME: refactor
            sequence_length: Maximum number of words per sequence.
            If max_words > 1 the first two dimensions of the output will be
                [batch_size, max_words].

            word_length: Number of characters per word to binarize. If
            the number of characters is above word_length it will be truncated.
            if word_length is below the number it will be padded. Defaults to
            16 which works well. . Note: if you are using
            a pretrained model you can't change this as it will break it.

            encoding_size: Size of output character encoding. Defaults to 32
            which ensure all printable code points can be perfectly
            represented. Can be lowered if needed but doesn't
            yield meaningful performance improvement. Note: if you are using
            a pretrained model you can't change this as it will break it.
        """
        super().__init__(**kwargs)
        self.word_length = word_length
        self.sequence_length = sequence_length
        self.encoding_size = encoding_size

        max_int32 = tf.constant([2**31], dtype="int64")
        bits_masks = tf.bitwise.right_shift(max_int32,
                                            tf.range(self.encoding_size,
                                                     dtype="int64"))
        bits_masks = tf.cast(bits_masks, dtype="int32")
        self.bits_masks = bits_masks

    def call(self, inputs: Tensor) -> Tensor:
        batch_size = tf.shape(inputs)[0]

        # Project into smaller space
        embeddings = self._project(inputs, self.bits_masks)

        # cast to float to be like a normal layer output
        embeddings = tf.cast(embeddings, dtype="float32")

        # reshape back to correct shape
        return _reshape_embeddings(embeddings, batch_size=batch_size,
                                   sequence_length=self.sequence_length,
                                   word_length=self.word_length,
                                   encoding_size=self.encoding_size)

    def _project(self, chars: Tensor, masks: Tensor) -> Tensor:
        """Project chars in subspace"""
        out = tf.bitwise.bitwise_and(tf.expand_dims(chars, -1), masks)
        out = tf.minimum(out, 1)
        return out

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "max_chars": self.word_length,
                "max_words": self.sequence_length,
                "encoding_size": self.encoding_size,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="retvec")
class RetVecBinarizer(tf.keras.layers.Layer):
    """RetVec binarizer which encodes all characters in the input
    into a compact binary representations.

    RetVec models are trained on top of this representation. This layer can
    also operate as a substitute for other unicode character encoding
    methodologies.

    Inputs to this model can be 1D (batch_size,) or 2D (batch_size, max_words).
    This layer supports both tf.Tensor and tf.RaggedTensor inputs.
    """

    def __init__(
        self,
        max_chars: int = 16,
        encoding_size: int = 32,
        encoding_type: str = "UTF-8",
        cls_int: Optional[int] = None,
        replacement_int: int = 11,
        allow_native: bool = False,
        **kwargs
    ) -> None:
        """Initialize a RetVec binarizer.

        Args:
            max_chars: Maximum number of characters to binarize. If the input
                is 2d, i.e. (batch_size, num_words) this is still the max
                characters per word.

            encoding_size: Size of output character encoding (in bits).

            encoding_type: String name for the unicode encoding that should
                be used to decode each string.

            cls_int: CLS token to prepend.

            replacement_int: The replacement integer to be used in place
                of invalid characters in input.

            allow_native: A boolean indicating whether to use
                `tensorflow_text.utf8_binarize` whenever possible
                (limited by its availability and constraints).

            **kwargs: Keyword args passed to the base Layer class.
        """
        super().__init__(**kwargs)

        self.max_chars = max_chars
        self.encoding_size = encoding_size
        self.encoding_type = encoding_type
        self.cls_int = cls_int
        self.replacement_int = replacement_int
        self.allow_native = allow_native

        # Check if the native `utf8_binarize` op is available for use.
        is_utf8_encoding = re.match('^utf-?8$', encoding_type, re.IGNORECASE)
        self._native_mode = (allow_native and
                             is_utf8_encoding and
                             utf8_binarize is not None and
                             cls_int is None)
        if allow_native and not self._native_mode:
            logging.warning('Native support for `RetVecBinarizer` unavailable. '
                            'Check `tensorflow_text.utf8_binarize` availability'
                            ' and its parameter contraints.')

        # Set to True when 'binarize()' is called in eager mode
        self.eager = False
        self._integerizer = None if self._native_mode else RetVecIntegerizer(
            max_chars=self.max_chars,
            encoding_type=self.encoding_type,
            cls_int=self.cls_int,
            replacement_int=self.replacement_int,
        )

    def build(self, input_shape: Union[TensorShape, List[TensorShape]]) -> None:
        self.max_words = input_shape[-1] if len(input_shape) > 1 else 1

        # Initialize int binarizer layer here since we know max_words
        self._int_binarizer = None if self._native_mode else RetVecIntToBinary(
            word_length=self.max_chars,
            sequence_length=self.max_words,
            encoding_size=self.encoding_size,
        )

    def call(self, inputs: Tensor) -> Tensor:
        if self._native_mode:
            embeddings = utf8_binarize(inputs,
                                       word_length=self.max_chars,
                                       bits_per_char=self.encoding_size,
                                       replacement_char=self.replacement_int)
            batch_size = tf.shape(inputs)[0]
            return _reshape_embeddings(embeddings, batch_size=batch_size,
                                       sequence_length=self.max_words,
                                       word_length=self.max_chars,
                                       encoding_size=self.encoding_size)
        else:
            assert self._integerizer is not None
            char_encodings = self._integerizer(inputs)
            assert self._int_binarizer is not None
            embeddings = self._int_binarizer(char_encodings)
            return embeddings

    def binarize(self, words: Tensor) -> Tensor:
        """Return RetVec binarizer encodings for a word or a list of words.

        Args:
            words: A single word or list of words to encode.

        Returns:
            Retvec binarizer encodings for the input words(s).
        """
        if words.shape == tf.TensorShape([]):
            inputs = tf.expand_dims(words, 0)
        else:
            inputs = words

        # set layers to eager mode
        self.eager = True
        if self._integerizer is not None:
            self._integerizer.eager = True

        # apply binarization
        embeddings = self(inputs)

        # Remove extra dim if input was a single word
        if words.shape == tf.TensorShape([]):
            embeddings = tf.reshape(embeddings[0], (self.max_chars, self.encoding_size))

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "max_chars": self.max_chars,
                "encoding_size": self.encoding_size,
                "encoding_type": self.encoding_type,
                "cls_int": self.cls_int,
                "replacement_int": self.replacement_int,
                "allow_native": self.allow_native,
            }
        )
        return config

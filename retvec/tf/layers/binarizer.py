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

from typing import Any, Dict, List, Union

import tensorflow as tf
from tensorflow import Tensor, TensorShape

from .integerizer import RETVecIntegerizer


@tf.keras.utils.register_keras_serializable(package="retvec")
class RETVecIntToBinary(tf.keras.layers.Layer):
    """Convert Unicode integer code points to their float binary encoding."""

    def __init__(
        self,
        sequence_length: int = 1,
        word_length: int = 16,
        encoding_size: int = 24,
        **kwargs
    ) -> None:
        """Initialize a RETVec int to binary converter.

        Args:
            sequence_length: Maximum number of words per sequence.
                If sequence_length > 1, the output will be reshaped to
                (`sequence_length`, `word_length`, `encoding_size`) for each
                element of the batch. Otherwise, if sequence_length is 1,
                the output will have shape (`word_length`, `encoding_size`).

            word_length: The number of characters per word to binarize.
                If the number of characters is above `word_length`, it will
                be truncated to `word_length` characters. If the number
                of characters is below `word_length`, it will be padded to
                `word_length`. Note: if you are using a pretrained RETVec
                model, `word_length` must match the length used in the model
                or else it will break.

            encoding_size: Size of output character encoding. Defaults to 24,
                which ensures that all UTF-8 codepoints can be uniquely
                represented. Note: if you are using a pretrained RETVec
                model, `encoding_size` must match the encoding size used in
                the model or else it will break.

            **kwargs: Additional keyword args passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.word_length = word_length
        self.sequence_length = sequence_length
        self.encoding_size = encoding_size

        max_int = tf.constant([2 ** (encoding_size - 1)], dtype="int64")
        bits_masks = tf.bitwise.right_shift(
            max_int, tf.range(encoding_size, dtype="int64")
        )
        bits_masks = tf.cast(bits_masks, dtype="int64")
        self.bits_masks = bits_masks

    def call(self, inputs: Tensor) -> Tensor:
        batch_size = tf.shape(inputs)[0]

        # Project into smaller space
        embeddings = self._project(inputs, self.bits_masks)

        # cast to float to be like a normal layer output
        embeddings = tf.cast(embeddings, dtype="float32")

        # reshape back to correct shape
        if self.sequence_length > 1:
            embeddings = tf.reshape(
                embeddings,
                (
                    batch_size,
                    self.sequence_length,
                    self.word_length,
                    self.encoding_size,
                ),
            )
        else:
            embeddings = tf.reshape(
                embeddings, (batch_size, self.word_length, self.encoding_size)
            )

        return embeddings

    def _project(self, chars: Tensor, masks: Tensor) -> Tensor:
        """Project chars in subspace"""
        chars = tf.cast(chars, dtype="int64")
        out = tf.bitwise.bitwise_and(tf.expand_dims(chars, -1), masks)
        out = tf.minimum(out, 1)
        return out

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "word_length": self.word_length,
                "sequence_length": self.sequence_length,
                "encoding_size": self.encoding_size,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="retvec")
class RETVecBinarizer(tf.keras.layers.Layer):
    """Transforms input text to a sequence of compact, binary representations
    for each Unicode character. Pretrained RETVec models are trained on top of
    this representation.

    Inputs to this model can be 1D (batches of single tokens) or 2D (batches of
    token sequences). This layer supports both tf.Tensor and tf.RaggedTensor
    inputs.
    """

    def __init__(
        self,
        word_length: int = 16,
        encoding_size: int = 24,
        encoding_type: str = "UTF-8",
        replacement_char: int = 65533,
        **kwargs
    ) -> None:
        """Initialize a RETVec binarizer.

        Args:
            word_length: The number of characters per word to binarize.
                If the number of characters is above `word_length`, it will
                be truncated to `word_length` characters. If the number
                of characters is below `word_length`, it will be padded to
                `word_length`. If the input is 2D, i.e. (batch_size,
                sequence_length) this is still the max characters per word.

                Note: if you are using a pretrained RETVec
                model, `word_length` must match the length used in the model
                or else it will break.

            encoding_size: Size of output character encoding. Defaults to 24,
                which ensures that all UTF-8 codepoints can be uniquely
                represented.

                Note: if you are using a pretrained RETVec
                model, `encoding_size` must match the encoding size used in
                the model or else it will break.

            encoding_type: String name for the unicode encoding that should
                be used to decode each string.

            replacement_char: The replacement Unicode integer codepoint to be
                used in place of invalid substrings in the input.

            **kwargs: Additional keyword args passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.word_length = word_length
        self.encoding_size = encoding_size
        self.encoding_type = encoding_type
        self.replacement_char = replacement_char

        # Set to True when 'binarize()' is called in eager mode
        self.eager = False
        self._integerizer = RETVecIntegerizer(
            word_length=self.word_length,
            encoding_type=self.encoding_type,
            replacement_char=self.replacement_char,
        )

    def build(
        self, input_shape: Union[TensorShape, List[TensorShape]]
    ) -> None:
        self.sequence_length = input_shape[-1] if len(input_shape) > 1 else 1

        # Initialize int binarizer layer here since we know sequence_length
        # only once we known the input_shape
        self._int_to_binary = RETVecIntToBinary(
            word_length=self.word_length,
            sequence_length=self.sequence_length,
            encoding_size=self.encoding_size,
        )

    def call(self, inputs: Tensor) -> Tensor:
        char_encodings = self._integerizer(inputs)
        embeddings = self._int_to_binary(char_encodings)
        return embeddings

    def binarize(self, inputs: Tensor) -> Tensor:
        """Return binary encodings for a word or a list of words.

        Args:
            inputs: A single word or list of words to encode.

        Returns:
            RETVec binary encodings for the input words(s).
        """
        inputs_shape = inputs.shape
        if inputs_shape == tf.TensorShape([]):
            inputs = tf.expand_dims(inputs, 0)

        # set layers to eager mode
        self.eager = True
        self._integerizer.eager = True

        # apply binarization
        embeddings = self(inputs)

        # Remove extra dim if input was a single word
        if inputs_shape == tf.TensorShape([]):
            embeddings = tf.reshape(
                embeddings[0], (self.word_length, self.encoding_size)
            )

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "word_length": self.word_length,
                "encoding_size": self.encoding_size,
                "encoding_type": self.encoding_type,
                "replacement_char": self.replacement_char,
            }
        )
        return config

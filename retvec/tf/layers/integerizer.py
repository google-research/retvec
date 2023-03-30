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


@tf.keras.utils.register_keras_serializable(package="retvec")
class RETVecIntegerizer(tf.keras.layers.Layer):
    """Transforms input text into a sequence of Unicode code points."""

    def __init__(
        self,
        word_length: int = 16,
        encoding_type: str = "UTF-8",
        replacement_char: int = 65533,
        **kwargs
    ) -> None:
        """Initialize a RetVec integerizer.

        Args:
            word_length: The number of characters per word to binarize.
                If the number of characters is above `word_length`, it will
                be truncated to `word_length` characters. If the number
                of characters is below `word_length`, it will be padded to
                `word_length`. Note: if you are using a pretrained RETVec
                model, `word_length` must match the length used in the model
                or else it will break.

            encoding_type: String name for the unicode encoding that should
                be used to decode each string.

            replacement_char: The replacement Unicode integer codepoint to be
                used in place of invalid substrings in the input.

            **kwargs: Additional keyword args passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.word_length = word_length
        self.encoding_type = encoding_type
        self.replacement_char = replacement_char
        self.eager = False  # Set to True if `binarize()` is called

    def build(
        self, input_shape: Union[TensorShape, List[TensorShape]]
    ) -> None:
        self.sequence_length = input_shape[-1]

        # We compute input rank here because rank must be statically known
        # for tf.string.unicode_decode. If the last dim is 1, we will
        # reshape it away later so rank - 1
        if input_shape[-1] == 1:
            self.input_rank = len(input_shape) - 1
        else:
            self.input_rank = len(input_shape)

    def call(self, chars: Tensor) -> Tensor:
        batch_size = tf.shape(chars)[0]

        # Reshape (and reshape back at the end) two dimensional inputs
        if self.input_rank == 2:
            chars = tf.reshape(chars, (batch_size * self.sequence_length, 1))

        # Apply unicode encoding to convert into integers
        char_codepoints = tf.strings.unicode_decode(
            chars,
            self.encoding_type,
            errors="replace",
            replacement_char=self.replacement_char,
        )

        # Handle shape differences between eager and graph mode
        if self.eager:
            if self.input_rank == 2:
                char_codepoints = tf.squeeze(char_codepoints, axis=1)
            char_codepoints = char_codepoints.to_tensor(
                shape=(char_codepoints.shape[0], self.word_length)
            )
        else:
            char_codepoints = char_codepoints.to_tensor(
                shape=(char_codepoints.shape[0], 1, self.word_length)
            )
            char_codepoints = tf.squeeze(char_codepoints, axis=1)

        # Reshape two dimensional inputs back
        if self.input_rank == 2:
            char_codepoints = tf.reshape(
                char_codepoints,
                (batch_size, self.sequence_length, self.word_length),
            )

        return char_codepoints

    def integerize(self, inputs: Tensor) -> Tensor:
        """Return Unicode integer code point encodings
        for a word or list of words.

        Args:
            inputs: A single word or list of words to encode.

        Returns:
            Retvec integerizer encodings for the input words(s).
        """
        inputs_shape = inputs.shape
        if inputs_shape == tf.TensorShape([]):
            inputs = tf.expand_dims(inputs, 0)

        # set to eager mode
        self.eager = True

        # apply integerization
        embeddings = self(inputs)

        # remove extra dim if input was a single word
        if inputs_shape == tf.TensorShape([]):
            embeddings = tf.reshape(embeddings[0], (self.word_length,))

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "word_length": self.word_length,
                "encoding_type": self.encoding_type,
                "replacement_char": self.replacement_char,
            }
        )
        return config

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

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow import Tensor, TensorShape


@tf.keras.utils.register_keras_serializable(package="retvec")
class RetVecIntegerizer(tf.keras.layers.Layer):
    """RetVec integerizer layer. This layer transforms string inputs
    into an integer representation (i.e. UTF-8 code points), which will
    then be passed into a binarizer.

    This layer currently only supports Unicode decodings.
    """

    def __init__(
        self,
        max_chars: int = 16,
        encoding_type: str = "UTF-8",
        cls_int: Optional[int] = None,
        replacement_int: int = 11,
        **kwargs
    ) -> None:
        """Initialize a RetVec integerizer.

        Args:
            max_chars: Maximum number of characters per word to integerize.

            encoding_type: String name for the unicode encoding that should
                be used to decode each string.

            cls_int: CLS token to prepend.

            replacement_int: The replacement integer to be used in place
                of invalid characters in input.
        """
        super().__init__(**kwargs)
        self.max_chars = max_chars
        self.encoding_type = encoding_type
        self.cls_int = cls_int
        self.replacement_int = replacement_int
        self.eager = False  # Set to True if `binarize()` is called

        if self.cls_int:
            self.pad_position = tf.constant([[0, 0], [1, 0]])
            self.pad_value = tf.constant(cls_int)

    def build(self, input_shape: Union[TensorShape, List[TensorShape]]) -> None:
        self.max_words = input_shape[-1]

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
            chars = tf.reshape(chars, (batch_size * self.max_words, 1))

        # Apply unicode encoding to convert into integers
        char_codepoints = tf.strings.unicode_decode(
            chars,
            self.encoding_type,
            errors="replace",
            replacement_char=self.replacement_int,
        )

        # Handle shape differences between eager and graph mode
        if self.eager:
            if self.input_rank == 2:
                char_codepoints = tf.squeeze(char_codepoints, axis=1)
            char_codepoints = char_codepoints.to_tensor(shape=(char_codepoints.shape[0], self.max_chars))
        else:
            char_codepoints = char_codepoints.to_tensor(shape=(char_codepoints.shape[0], 1, self.max_chars))
            char_codepoints = tf.squeeze(char_codepoints, axis=1)

        # add CLS int and then reshape to max size
        if self.cls_int:
            char_codepoints = tf.pad(char_codepoints, self.pad_position, constant_values=self.pad_value)
            char_codepoints = char_codepoints[:, : self.max_chars]

        # Reshape two dimensional inputs back
        if self.input_rank == 2:
            char_codepoints = tf.reshape(char_codepoints, (batch_size, self.max_words, self.max_chars))

        return char_codepoints

    def integerize(self, words: Tensor) -> Tensor:
        """Return RetVec integerizer encodings for a word or a list of words.

        Args:
            words: A single word or list of words to encode.

        Returns:
            Retvec integerizer encodings for the input words(s).
        """
        if words.shape == tf.TensorShape([]):
            inputs = tf.expand_dims(words, 0)
        else:
            inputs = words

        # set to eager mode
        self.eager = True

        # apply integerization
        embeddings = self(inputs)

        # remove extra dim if input was a single word
        if words.shape == tf.TensorShape([]):
            embeddings = tf.reshape(embeddings[0], (self.max_chars,))

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "max_chars": self.max_chars,
                "encoding_type": self.encoding_type,
                "cls_int": self.cls_int,
                "replacement_int": self.replacement_int,
            }
        )
        return config

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

from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers

from .binarizers import RetVecBinarizer
from .embedding import RetVecEmbedding


@tf.keras.utils.register_keras_serializable(package="retvec")
class RetVec(tf.keras.layers.Layer):
    """RetVec: Resilient and Efficient Vectorizer layer."""

    def __init__(
        self,
        max_len: int = 128,
        sep: str = "",
        lowercase: bool = False,
        model: Optional[str] = None,
        trainable: bool = False,
        max_chars: int = 16,
        char_encoding_size: int = 32,
        char_encoding_type: str = "UTF-8",
        cls_int: Optional[int] = None,
        replacement_int: int = 11,
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        norm_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize a RetVec layer.

        Args:
            max_len: Max length of text split by `sep`.

            sep: Separator to split input text on into words. Defaults to '',
                which splits on all whitespace and stripts whitespace from
                beginning or end of the text.

            lowercase: Whether to lowercase input text.

            model: Path to saved REW* model. If None, this layer will default
                to the RetVecBinarizer character encoding.

            trainable: Whether or not this layer should be trainable.

            max_chars: Maximum number of characters per word to integerize.

            char_encoding_size: Size of the character encoding per word.

            char_encoding_type: String name for the unicode encoding that
                should be used to decode each string.

            cls_int: CLS token to prepend to each word encoding.

            replacement_int: The replacement integer to be used in place
                of invalid characters in input.

            dropout_rate: Dropout to apply after RetVec embedding.

            spatial_dropout_rate: Spatial dropout to apply after RetVec
                embedding.

            norm_type: Norm to apply after RetVec embedding. One of
                [None, 'batch', or 'layer'].

        """
        super().__init__(**kwargs)

        self.max_len = max_len
        self.sep = sep
        self.lowercase = lowercase
        self.model = model
        self.trainable = trainable

        # RetVecEmbedding
        if self.model:
            self._embedding: Optional[RetVecEmbedding] = RetVecEmbedding(model=model, trainable=self.trainable)
        else:
            self._embedding = None

        # RetVecBinarizer
        self.max_chars = max_chars
        self.char_encoding_size = char_encoding_size
        self.char_encoding_type = char_encoding_type
        self.cls_int = cls_int
        self.replacement_int = replacement_int
        self._binarizer = RetVecBinarizer(
            max_chars=self.max_chars,
            encoding_size=self.char_encoding_size,
            encoding_type=self.char_encoding_type,
            cls_int=self.cls_int,
            replacement_int=self.replacement_int,
        )

        # Set to True when 'tokenize()' or 'binarize()' called in eager mode
        self.eager = False

        if self._embedding:
            self._embedding_size = self._embedding.embedding_size
        else:
            self._embedding = None
            self._embedding_size = self.max_chars * self.char_encoding_size

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

    @tf.function()
    def call(self, inputs: Tensor, training: bool = False) -> Tensor:
        inputs = tf.stop_gradient(inputs)

        if self.lowercase:
            inputs = tf.strings.lower(inputs)

        rtensor = tf.strings.split(inputs, sep=self.sep, maxsplit=self.max_len)

        # Handle shape differences between eager and graph mode
        if self.eager:
            stensor = rtensor.to_tensor(default_value="", shape=(rtensor.shape[0], self.max_len))
        else:
            stensor = rtensor.to_tensor(default_value="", shape=(rtensor.shape[0], 1, self.max_len))
            stensor = tf.squeeze(stensor, axis=1)

        # apply encoding and REW* model, if set
        binarized = self._binarizer(stensor, training=training)

        if self._embedding:
            embeddings = self._embedding(binarized, training=training)
        else:
            embsize = self._binarizer.encoding_size * self._binarizer.max_chars
            embeddings = tf.reshape(binarized, (tf.shape(inputs)[0], self.max_len, embsize))

        # apply post-embedding norm and dropout layers
        if self.norm_type:
            embeddings = self.norm(embeddings, training=training)

        if self.dropout_rate:
            embeddings = self.dropout(embeddings, training=training)

        if self.spatial_dropout_rate:
            embeddings = self.spatial_drop(embeddings, training=training)

        return embeddings

    @tf.function()
    def binarize(self, words: Tensor) -> Tensor:
        """Return RetVec binarizer encodings for a word or a list of words.

        Args:
            words: A single word or list of words to encode.

        Returns:
            Retvec binarizer encodings for the input words(s).
        """
        return self._binarizer.binarize(words)

    @tf.function()
    def tokenize(self, words: Tensor) -> Tensor:
        """Return RetVec embeddings for a word or a list of words.

        Args:
            words: A single word or list of words to encode.

        Returns:
            Retvec embeddings for the input words(s).
        """
        if words.shape == tf.TensorShape([]):
            inputs = tf.expand_dims(words, 0)
        else:
            inputs = words

        # set layers to eager mode
        self.eager = True
        self._binarizer.eager = True

        # compute embeddings
        embeddings = self(inputs, training=False)

        # Remove extra dim if input was a single word
        if words.shape == tf.TensorShape([]):
            embeddings = tf.squeeze(embeddings)

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "sep": self.sep,
                "lowercase": self.lowercase,
                "model": self.model,
                "trainable": self.trainable,
                "max_chars": self.max_chars,
                "char_encoding_size": self.char_encoding_size,
                "char_encoding_type": self.char_encoding_type,
                "cls_int": self.cls_int,
                "replacement_int": self.replacement_int,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "norm_type": self.norm_type,
            }
        )
        return config

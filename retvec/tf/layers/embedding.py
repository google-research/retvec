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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow import Tensor, TensorShape


@tf.keras.utils.register_keras_serializable(package="retvec")
class RETVecEmbedding(tf.keras.layers.Layer):
    """Applies a pretrained RETVec word embedding model to binarized inputs
    to generate word embeddings."""

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        trainable: bool = False,
        **kwargs
    ) -> None:
        """Initialize a RETVecEmbedding layer.

        Args:
            model: Path to saved pretrained RETVec model, str or pathlib.Path
                object.

            trainable: Whether to make the pretrained RETVec model trainable
                or to freeze all weights.

            **kwargs: Additional keyword args passed to the base Layer class.
        """
        if not model:
            raise ValueError("`model` must be set for RETVecEmbedding layer.")

        super().__init__(**kwargs)
        self.model = model
        self.trainable = trainable

        # Load REW* model
        self.rewnet = self._load_model(model)
        self.embedding_size = self.rewnet.layers[-1].output_shape[-1]

    def build(
        self, input_shape: Union[TensorShape, List[TensorShape]]
    ) -> None:
        self.input_rank = len(input_shape)

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:
        input_shape = tf.shape(inputs)

        # Reshape inputs like (batch_size, max_words, max_chars, encoding_size)
        if self.input_rank == 4:
            batch_size = input_shape[0]
            num_words = input_shape[1]
            max_chars = input_shape[2]
            encoding_size = input_shape[-1]
            inputs = tf.reshape(
                inputs, (batch_size * num_words, max_chars, encoding_size)
            )
        else:
            batch_size = input_shape[0]
            max_chars = input_shape[1]
            encoding_size = input_shape[2]
            num_words = tf.constant(1, dtype=tf.int32)

        output = self.rewnet(inputs, training=training)

        # Reshape inputs back if needed
        if self.input_rank == 4:
            output = tf.reshape(
                output, (batch_size, num_words, self.embedding_size)
            )

        return output

    def _load_model(
        self, path: Optional[Union[str, Path]] = None
    ) -> tf.keras.models.Model:
        """Load pretrained RETVec model.

        Args:
            path: Path to the saved REW* model.

        Returns:
            The pretrained RETVec model, trainable set to `self.trainable`.
        """
        model = tf.keras.models.load_model(path)
        model.trainable = self.trainable
        model.compile("adam", "mse")
        return model

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update({"model": self.model, "trainable": self.trainable})
        return config

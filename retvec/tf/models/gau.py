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

from typing import Any, Dict

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from ..utils import clone_initializer
from .layers import get_activation_layer, get_norm_layer
from .positional_embeddings import toeplitz_matrix, toeplitz_matrix_rope

ZEROS_INTIALIZER = tf.initializers.zeros()


@tf.keras.utils.register_keras_serializable(package="retvec")
class GAU(Layer):
    """Gated Attention Unit layer introduced in Transformer
    Quality in Linear Time.

    Paper reference: https://arxiv.org/abs/2202.10447
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 128,
        shared_dim: int = 128,
        expansion_factor: int = 2,
        activation: str = "swish",
        attention_activation: str = "sqrrelu",
        norm_type: str = "scaled",
        position_encoding_type: str = "rope",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        epsilon: float = 1e-6,
        **kwargs
    ) -> None:
        """
        Initialize a GAU layer.

        Args:
            dim: Dimension of GAU block.

            max_len: Maximum seq len of input. Defaults to 128.

            shared_dim: Size of shared dim. Defaults to 128.

            expansion_factor: Hidden dim expansion factor. Defaults to 2.

            activation: Activation to use in projection layers. Defaults
                to 'swish'.

            attention_activation: Activation to use on attention scores.
                Defaults to 'sqrrelu'.

            norm_type: Norm type. One of 'layer', 'scaled', 't5' or None.
                Defaults to 'scaled'.

            position_encoding_type: Type of positional encoding to use.
                Either 'rope' or 'relative'. Defaults to 'rope'.

            dropout_rate: Feature dropout rate. Defaults to 0.0.

            attention_dropout_rate: Attention dropout rate.
                Defaults to 0.0.

            spatial_dropout_rate: Spatial dropout rate. Defaults to 0.0.

            epsilon: Epsilon value for norm. Defaults to 1e-6.
        """
        super().__init__(**kwargs)

        self.dim = dim
        self.max_len = max_len
        self.shared_dim = shared_dim
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.attention_activation = attention_activation
        self.norm_type = norm_type
        self.position_encoding_type = position_encoding_type
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.epsilon = epsilon

        # compute projection dimension
        self.expand_dim = self.dim * self.expansion_factor
        self.proj_dim = 2 * self.expand_dim + self.shared_dim
        self.weight_initializer = tf.random_normal_initializer(stddev=0.02)

        # define layers
        self.norm = get_norm_layer(norm=self.norm_type, epsilon=self.epsilon)

        self.proj1 = layers.Dense(
            self.proj_dim,
            use_bias=True,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.weight_initializer),
            bias_initializer="zeros",
        )
        self.proj2 = layers.Dense(
            self.dim,
            use_bias=True,
            kernel_initializer=clone_initializer(self.weight_initializer),
            bias_initializer="zeros",
        )

        # dropout layers
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)

        if self.attention_dropout_rate:
            self.attention_dropout = layers.Dropout(
                self.attention_dropout_rate
            )

        if self.spatial_dropout_rate:
            self.spatial_dropout = layers.SpatialDropout1D(
                self.spatial_dropout_rate
            )

        # attention activation function
        self.attention_activation_layer = get_activation_layer(
            attention_activation
        )

        # setting up position encoding
        if self.position_encoding_type == "relative":
            self.w = tf.Variable(
                lambda: clone_initializer(
                    self.weight_initializer(
                        shape=[2 * self.max_len - 1], dtype=tf.float32
                    )
                ),
                trainable=True,
            )

        elif self.position_encoding_type == "rope":
            self.a = tf.Variable(
                lambda: clone_initializer(self.weight_initializer)(
                    shape=[self.max_len], dtype=tf.float32
                ),
                trainable=True,
            )
            self.b = tf.Variable(
                lambda: clone_initializer(self.weight_initializer)(
                    shape=[self.max_len], dtype=tf.float32
                ),
                trainable=True,
            )

        # offset scaling values
        self.gamma = tf.Variable(
            lambda: clone_initializer(self.weight_initializer)(
                shape=[2, self.shared_dim], dtype=tf.float32
            ),
            trainable=True,
        )

        self.beta = tf.Variable(
            lambda: ZEROS_INTIALIZER(
                shape=[2, self.shared_dim], dtype=tf.float32
            ),
            trainable=True,
        )

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        shortcut = x
        x = self.norm(x)

        # input dropout
        if self.spatial_dropout_rate:
            x = self.spatial_dropout(x, training=training)

        x = self.dropout1(x, training=training)

        # initial projection to generate uv
        uv = self.proj1(x)
        uv = self.dropout2(uv, training=training)

        u, v, base = tf.split(
            uv, [self.expand_dim, self.expand_dim, self.shared_dim], axis=-1
        )

        # generate q, k by scaled offset using TF-Lite compatible ops instead of einsum
        # base = tf.einsum("bnr,hr->bnhr", base, self.gamma) + self.beta
        base = (
            tf.tile(tf.expand_dims(base, 2), [1, 1, 2, 1]) * self.gamma
            + self.beta
        )
        q, k = tf.unstack(base, axis=-2)

        # compute key-query scores
        qk = tf.einsum("bnd,bmd->bnm", q, k)
        qk = qk / self.max_len

        # add relative position bias for attention
        if self.position_encoding_type == "relative":
            bias = toeplitz_matrix(self.max_len, self.w)
            qk += bias

        elif self.position_encoding_type == "rope":
            bias = toeplitz_matrix_rope(self.max_len, self.a, self.b)
            qk += bias

        # apply attention activation and dropout
        kernel = self.attention_activation_layer(qk)

        if self.attention_dropout_rate:
            kernel = self.attention_dropout(kernel)

        # apply values and project
        x = u * tf.einsum("bnm,bme->bne", kernel, v)
        x = self.proj2(x)

        return x + shortcut

    def get_config(self) -> Dict[str, Any]:
        config: Dict = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "max_len": self.max_len,
                "shared_dim": self.shared_dim,
                "expansion_factor": self.expansion_factor,
                "activation": self.activation,
                "attention_activation": self.attention_activation,
                "norm_type": self.norm_type,
                "position_encoding_type": self.position_encoding_type,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
                "epsilon": self.epsilon,
            }
        )
        return config

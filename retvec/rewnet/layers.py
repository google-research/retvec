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
from retvec.types import FloatTensor
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


def dense_block(x: FloatTensor,
                units: int,
                activation: str,
                batch_norm: bool = True) -> FloatTensor:
    x = layers.Dense(units)(x)

    if batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Activation(activation)(x)
    return x


@tf.keras.utils.register_keras_serializable(package="retvec")
class L2Norm(layers.Layer):
    def __init__(self, **kwargs) -> None:
        """L2 Normalization layer usually used as output layer.
        """
        super().__init__(**kwargs)

    def call(self, inputs: FloatTensor) -> FloatTensor:
        normed_x: FloatTensor = tf.math.l2_normalize(inputs, axis=1)
        return normed_x


@tf.keras.utils.register_keras_serializable(package="retvec")
def sqrrelu(x):
    return tf.math.square(tf.nn.relu(x))


@tf.keras.utils.register_keras_serializable(package="retvec")
class FFN(Layer):
    def __init__(self,
                 hidden_size: int,
                 out_size: int,
                 activation: str,
                 dropout_rate: float = 0,
                 **kwargs) -> None:
        """
        Construct a standard FFN layer
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.hidden = layers.Dense(hidden_size, use_bias=False)
        self.activation = layers.Activation(activation)
        self.out = layers.Dense(out_size, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs: FloatTensor, training: bool = False) -> FloatTensor:
        inputs = self.hidden(inputs)
        inputs = self.activation(inputs)
        inputs = self.dropout(inputs, training=training)
        inputs = self.out(inputs)
        return inputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "out_size": self.out_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
class GatedFFN(Layer):
    def __init__(self,
                 hidden_size: int,
                 out_size: int,
                 activation: str,
                 dropout_rate: float = 0,
                 **kwargs) -> None:
        """Implements Gated FFN based off https://arxiv.org/pdf/2002.05202.pdf

        Note:
        - to be size equivalent, the hidden_dim should be about 2/3 of the
        standard FeedForward network
        - Swish activated gate and GELU activated gate seems to perform
        the best.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.Hidden = layers.Dense(hidden_size,  use_bias=False)
        self.Gate = layers.Dense(hidden_size,  use_bias=False)
        self.Activation = layers.Activation(activation)
        self.Out = layers.Dense(out_size, use_bias=False)
        self.Dropout = layers.Dropout(dropout_rate)

    def call(self, inputs: FloatTensor, training: bool = False) -> FloatTensor:

        # compute gate
        gate = self.Gate(inputs)
        gate = self.Activation(gate)

        # expand & gate
        hidden = self.Hidden(inputs)
        hidden = hidden * gate  # apply gate

        # drop & compress
        hidden = self.Dropout(hidden, training=training)
        hidden = self.Out(hidden)

        return hidden

    def get_config(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "out_size": self.out_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
class TBlock(layers.Layer):
    """
    Vanilla Transformer Block
    """

    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 out_dim: int = None,
                 head_dim: int = 64,
                 dropout_rate: float = 0.05,
                 spatial_dropout_rate: float = 0.05,
                 activation: str = 'gelu',
                 epsilon: float = 1e-6,
                 use_gated_ffn: bool = False,
                 use_cpe: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim if out_dim else self.dim
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.activation = activation
        self.epsilon = epsilon
        self.use_cpe = use_cpe
        self.use_gated_ffn = use_gated_ffn

        if use_cpe:
            self.cpe = layers.DepthwiseConv1D(kernel_size=1, strides=1)

        self.spatial_drop = layers.SpatialDropout1D(spatial_dropout_rate)

        self.norm1 = layers.LayerNormalization(epsilon=epsilon)
        self.norm2 = layers.LayerNormalization(epsilon=epsilon)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

        self.attention = layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=head_dim,
                                                   dropout=self.dropout_rate)

        if self.use_gated_ffn:
            self.ffn = GatedFFN(hidden_size=self.hidden_dim,
                                out_size=self.out_dim,
                                activation=self.activation,
                                dropout_rate=self.dropout_rate)
        else:
            self.ffn = FFN(hidden_size=self.hidden_dim,
                           out_size=self.out_dim,
                           activation=self.activation,
                           dropout_rate=self.dropout_rate)

    def call(self, inputs: FloatTensor, training: bool) -> FloatTensor:
        residual = inputs
        x = inputs

        if self.use_cpe:
            x = self.cpe(x)
            x = x + residual

        residual = x

        if self.spatial_dropout_rate:
            x = self.spatial_drop(x, training=training)

        x = self.dropout1(x, training=training)
        x = self.attention(x, x)
        x = self.dropout2(x, training=training)

        x = x + residual
        x = self.norm1(x, training=training)

        residual = x

        x = self.ffn(x, training=training)
        x = self.dropout3(x, training=training)

        if self.out_dim == self.dim:
            x = x + residual

        x = self.norm2(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "out_dim": self.out_dim,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "epsilon": self.epsilon,
            "use_cpe": self.use_cpe,
            "use_gated_ffn": self.use_gated_ffn,
            "spatial_dropout_rate": self.spatial_dropout_rate
        }

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

from typing import Any, Callable, Dict, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


def dense_block(
    x: Tensor,
    units: int,
    activation: str,
    norm_type: str = "batch",
    norm_epsilon: float = 1e-6,
    **dense_kwargs,
) -> Tensor:
    """Build a dense block for a TF model."""
    x = layers.Dense(units, **dense_kwargs)(x)

    if norm_type:
        x = get_norm_layer(norm=norm_type, epsilon=norm_epsilon)(x)

    x = get_activation_layer(activation)(x)
    return x


def get_activation_layer(activation: str):
    """Get activation layer or function for TF model."""
    if activation == "sqrrelu":
        return sqrrelu
    elif activation == "sin":
        return tf.sin
    elif activation == "cos":
        return tf.cos
    elif activation == "id":
        return lambda x: x
    else:
        return tf.keras.layers.Activation(activation)


def get_norm_layer(norm: str, **kwargs):
    """Get normalization layer for TF model."""
    if norm == "layer":
        return layers.LayerNormalization(**kwargs)
    elif norm == "batch":
        return layers.BatchNormalization(**kwargs)
    elif norm == "scaled":
        return ScaledNorm(**kwargs)
    elif norm == "l2":
        return L2Norm(**kwargs)
    else:
        raise ValueError(f"Unsupported norm type {norm}")


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
class L2Norm(layers.Layer):
    """L2 Normalization layer usually used as output layer."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        normed_x: Tensor = tf.math.l2_normalize(inputs, axis=1)
        return normed_x


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
def sqrrelu(x: Tensor) -> Tensor:
    """Squared ReLU (ReLU(x) ** 2).

    Args:
      v: The input tensor to sqrrelu.

    Returns:
        Second power of ReLU(x).
    """
    return tf.math.square(tf.nn.relu(x))


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
class BertPooling(Layer):
    """Bert pooling layer."""

    def call(self, x: Tensor) -> Tensor:
        return x[:, 0]


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
class ScaledNorm(Layer):
    """ScaledNorm layer."""

    def __init__(self, begin_axis: int = -1, epsilon: float = 1e-5, **kwargs) -> None:
        """Initialize a ScaledNorm Layer.

        Args:
            begin_axis: Axis along which to apply norm. Defaults to -1.
            epsilon: Norm epsilon value. Defaults to 1e-5.
        """
        super().__init__(**kwargs)
        self._begin_axis = begin_axis
        self._epsilon = epsilon
        self._scale = self.add_weight(
            name="norm_scale",
            shape=(),
            initializer=tf.constant_initializer(value=1.0),
            trainable=True,
        )

    def call(self, inputs: Tensor) -> Tensor:
        x = inputs
        axes = list(range(len(x.shape)))[self._begin_axis :]
        mean_square = tf.reduce_mean(tf.math.square(x), axes, keepdims=True)
        x = x * tf.math.rsqrt(mean_square + self._epsilon)
        return x * self._scale

    def get_config(self) -> Dict[str, Any]:
        config = {"begin_axis": self._begin_axis, "epsilon": self._epsilon}
        base_config = super(ScaledNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
class ConvNextBlock(Layer):
    """ConvNeXt block.

    Adapted from A ConvNet for the 2020s (https://arxiv.org/pdf/2201.03545.pdf)

    This layer is compatitible with existing TensorFlow.js supported ops,
    which means that models built using this layer be converted to javascript
    using the TensorFlow.js converter. For more info, visit
    https://www.tensorflow.org/js/guide/conversion.
    """

    def __init__(
        self,
        kernel_size: int,
        depth: int,
        filters: int,
        hidden_dim: int,
        dropout_rate: float = 0,
        epsilon: float = 1e-10,
        activation: Union[str, Callable] = "gelu",
        strides: int = 1,
        residual: bool = True,
        **kwargs,
    ) -> None:
        """Initialize a ConvNextBlock.

        Args:
            kernel_size: Kernel size for convolution.

            depth: Depth multiplier for depthwise 1D convolution.

            filters: Number of convolution filters.

            hidden_dim: Hidden dim of block.

            dropout_rate: Feature dropout rate. Defaults to 0.

            epsilon: Layer norm epsilon. Defaults to 1e-10.

            activation: Layer activation. Defaults to 'gelu'.

            strides: Strides to apply convolution. Defaults to 1.

            residual: Whether to add residual connection. Defaults to True.
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth = depth
        self.filters = filters
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.activation = activation
        self.strides = strides
        self.residual = residual
        self.depthconv = layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            strides=strides,
            depth_multiplier=depth,
            padding="same",
        )
        self.norm = layers.LayerNormalization(epsilon=epsilon)
        self.hidden = layers.Dense(hidden_dim)
        self.activation = layers.Activation(activation)
        self.drop = layers.Dropout(dropout_rate)
        self.out = layers.Dense(filters)

    def call(self, inputs: Tensor, training: bool) -> Tensor:
        residual = inputs
        x = self.depthconv(inputs)
        x = self.norm(x, training=training)
        x = self.hidden(x)
        x = self.drop(x, training=training)
        x = self.out(x)
        if self.residual:
            x = x + residual
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "kernel_size": self.kernel_size,
            "depth": self.depth,
            "filters": self.filters,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "epsilon": self.epsilon,
            "activation": self.activation,
            "strides": self.strides,
            "residual": self.residual,
        }

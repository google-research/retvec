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
from tensorflow.keras.layers import Layer
from tensorflow_similarity.layers import GeneralizedMeanPooling1D


def dense_block(
    x: Tensor,
    units: int,
    activation: Optional[str] = None,
    norm_type: Optional[str] = None,
    norm_epsilon: float = 1e-6,
    dropout_rate: float = 0.0,
    spatial_dropout_rate: float = 0.0,
    name: Optional[str] = None,
    **dense_kwargs,
) -> Tensor:
    """Build a dense block for a TF model."""
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    if spatial_dropout_rate:
        x = layers.SpatialDropout1D(spatial_dropout_rate)(x)

    x = layers.Dense(units, **dense_kwargs)(x)

    if norm_type:
        x = get_norm_layer(norm=norm_type, epsilon=norm_epsilon)(x)

    x = get_activation_layer(activation, name=name)(x)
    return x


def get_activation_layer(activation: Optional[str] = None, **kwargs):
    """Get activation layer or function for TF model."""
    if activation == "sqrrelu":
        return SqrReLU(**kwargs)
    elif activation == "relu1":
        return tf.keras.layers.ReLU(max_value=1.0, **kwargs)
    elif activation == "relu2":
        return tf.keras.layers.ReLU(max_value=2.0, **kwargs)
    else:
        return tf.keras.layers.Activation(activation, **kwargs)


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


def get_pooling_layer(x: Tensor, pooling_type: str) -> Tensor:
    if pooling_type == "GEM":
        x = GeneralizedMeanPooling1D()(x)

    elif pooling_type == "max":
        x = layers.GlobalMaxPool1D()(x)

    elif pooling_type == "avg":
        x = layers.GlobalAveragePooling1D()(x)

    elif pooling_type == "flatten":
        x = layers.Flatten()(x)

    elif pooling_type == "bert":
        x = BertPooling()(x)

    return x


@tf.keras.utils.register_keras_serializable(package="retvec")
class L2Norm(layers.Layer):
    """L2 Normalization layer usually used as output layer."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        normed_x: Tensor = tf.math.l2_normalize(inputs, axis=1)
        return normed_x


@tf.keras.utils.register_keras_serializable(package="retvec")
class SqrReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SqrReLU, self).__init__(**kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        return tf.math.square(tf.nn.relu(inputs))


@tf.keras.utils.register_keras_serializable(package="retvec")
class BertPooling(Layer):
    """Bert pooling layer."""

    def call(self, inputs: Tensor) -> Tensor:
        return inputs[:, 0]


@tf.keras.utils.register_keras_serializable(package="retvec")
class ScaledNorm(Layer):
    """ScaledNorm layer."""

    def __init__(
        self, begin_axis: int = -1, epsilon: float = 1e-5, **kwargs
    ) -> None:
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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

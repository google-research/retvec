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

from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


@tf.keras.utils.register_keras_serializable(package="retvec")
class PositionalEmbedding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(
        self,
        min_timescale: float = 1.0,
        max_timescale: float = 1.0e4,
        **kwargs,
    ):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["min_timescale"] = self.min_timescale
        config["max_timescale"] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale
        )
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


def positional_signal(
    hidden_size: int,
    length: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1e4,
):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}"
        )
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (
            np.log(float(max_timescale) / float(min_timescale))
            / (num_timescales - 1)
        ),
        dtype=K.floatx(),
    )
    inv_timescales = min_timescale * K.exp(
        K.arange(num_timescales, dtype=K.floatx()) * -log_timescale_increment
    )
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


@tf.keras.utils.register_keras_serializable(package="retvec")
class ScaledSinusoidalPositionalEmbedding(Layer):
    """Creates a positional embedding with a learnable scalar for stability.

    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and
    formulized in "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).
    """

    def __init__(
        self,
        hidden_size: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1.0e4,
        **kwargs,
    ):
        """Initialize a ScaledSinusoidalPositionalEmbedding layer.

        Args:

            hidden_size: Size of the hidden layer.

            min_timescale: Minimum scale that will be applied at each position.

            max_timescale: Maximum scale that will be applied at each position.
        """
        # We need to have a default dtype of float32, since the inputs (which
        # Keras usually uses to infer the dtype) will always be int32.
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically
        # unstable in float16.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super().__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale
        self._init_scale = 1 / self._hidden_size**0.5

        self._scale = self.add_weight(
            name="sin_scale",
            shape=(),
            initializer=tf.constant_initializer(value=self._init_scale),
            trainable=True,
        )

    def get_config(self):
        config = {
            "hidden_size": self._hidden_size,
            "min_timescale": self._min_timescale,
            "max_timescale": self._max_timescale,
        }
        base_config = super(
            ScaledSinusoidalPositionalEmbedding, self
        ).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        length = inputs.shape[1]
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale

        # compute sinusoidal pos encodings
        log_timescale_increment = tf.math.log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32)
            * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0
        )
        position_embeddings = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1
        )

        # scale pos encodings with a learnable scalar
        position_embeddings = position_embeddings * self._scale

        return inputs + position_embeddings


@tf.keras.utils.register_keras_serializable(package="retvec")
def rope(x: Tensor, axis: Union[List[int], int]) -> Tensor:
    """RoPE positional encoding.

    Implementation of the Rotary Position Embedding proposed in
    https://arxiv.org/abs/2104.09864.

    Args:
        x: input tensor.
        axis: axis to add the positional encodings.

    Returns:
        The input tensor with RoPE encodings.
    """
    shape = x.shape.as_list()

    if isinstance(axis, int):
        axis = [axis]

    if isinstance(shape, (list, tuple)):
        spatial_shape = [shape[i] for i in axis]
        total_len = 1
        for i in spatial_shape:
            total_len *= i
        position = tf.reshape(
            tf.cast(tf.range(total_len, delta=1.0), tf.float32), spatial_shape
        )
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # we assume that the axis can not be negative (e.g., -1)
    if any(dim < 0 for dim in axis):
        raise ValueError(f"Unsupported axis: {axis}")
    for i in range(axis[-1] + 1, len(shape) - 1, 1):
        position = tf.expand_dims(position, axis=-1)

    half_size = shape[-1] // 2
    freq_seq = tf.cast(tf.range(half_size), tf.float32) / float(half_size)
    inv_freq = 10000**-freq_seq
    sinusoid = tf.einsum("...,d->...d", position, inv_freq)
    sin = tf.cast(tf.sin(sinusoid), dtype=x.dtype)
    cos = tf.cast(tf.cos(sinusoid), dtype=x.dtype)
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


@tf.keras.utils.register_keras_serializable(package="retvec")
def toeplitz_matrix_rope(n: int, a: Tensor, b: Tensor) -> Tensor:
    """Obtain Toeplitz matrix using rope."""
    a = rope(tf.tile(a[None, :], [n, 1]), axis=0)
    b = rope(tf.tile(b[None, :], [n, 1]), axis=0)
    return tf.einsum("mk,nk->mn", a, b)


@tf.keras.utils.register_keras_serializable(package="retvec")
def toeplitz_matrix(n: int, w: Tensor) -> Tensor:
    """Toeplitz matrix of shape [num_heads, n, n] or [n, n]."""
    paddings = [[0, n]]
    multiples = [n]
    t = tf.pad(w, paddings)
    t = tf.tile(t, multiples)
    t = t[..., :-n]
    t = tf.reshape(t, [n, 3 * n - 2])
    r = (2 * n - 1) // 2
    t = t[..., r:-r]
    return t

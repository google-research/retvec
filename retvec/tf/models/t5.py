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

import math
from typing import Any, Dict, List

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.python.framework.tensor_shape import TensorShape

from .layers import FFN, GatedFFN


@tf.keras.utils.register_keras_serializable(package="retvec")
class T5LayerNorm(Layer):
    def __init__(self, epsilon: float = 1e-6, **kwargs) -> None:
        """
        Construct a layernorm module in the T5 style.
        No bias and no subtraction of mean.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape: TensorShape) -> None:
        """Build shared word embedding layer"""
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)

    def call(self, inputs: Tensor) -> Tensor:
        variance = tf.math.square(inputs)
        variance = tf.math.reduce_mean(variance, axis=-1, keepdims=True)
        variance = tf.math.rsqrt(variance + self.epsilon)
        inputs = inputs * variance
        inputs = self.weight * inputs
        return inputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
        }


def shape_list(tensor: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor: The tensor we want the shape of.

    Returns:
        The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


@tf.keras.utils.register_keras_serializable(package="retvec")
class T5Attention(tf.keras.layers.Layer):
    """T5 attention mechanism with relative position.
    https://github.com/huggingface/transformers/blob/12b4d66a80419db30a15e7b9d4208ceb9887c03b/src/transformers/models/t5/modeling_tf_t5.py#L148
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int = 64,
        dropout_rate: float = 0,
        position_buckets: int = 32,
        position_max_distance: int = 128,
        position_bidirectional: bool = True,
        output_attention: bool = False,
        has_relative_attention_bias: bool = False,
        use_cache: bool = False,
        is_decoder: bool = False,  # we never are -- legacy
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # dimension
        self.dim = dim
        self.head_dim = head_dim  # formally kv dim (page 11)
        self.inner_dim = num_heads * head_dim

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # relative attention
        self.position_buckets = position_buckets
        self.position_max_distance = position_max_distance
        self.position_bidirectional = position_bidirectional

        # others
        self.output_attention = output_attention
        self.relative_attention_bias = None
        self.has_relative_attention_bias = has_relative_attention_bias
        self.use_cache = use_cache
        self.is_decoder = is_decoder

        # layers
        self.q = layers.Dense(self.inner_dim, use_bias=False, name="q")
        self.k = layers.Dense(self.inner_dim, use_bias=False, name="k")
        self.v = layers.Dense(self.inner_dim, use_bias=False, name="v")
        self.o = layers.Dense(self.dim, use_bias=False, name="o")
        self.dropout = layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        if self.has_relative_attention_bias:
            with tf.name_scope("relative_attention_bias"):
                shape = [self.position_buckets, self.num_heads]
                self.relative_attention_bias = self.add_weight(name="embeddings", shape=shape)

        return super().build(input_shape)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Adapted from Mesh tensorflow
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative
        attention. The relative position is defined as memory_position
        - query_position, i.e. the distance in tokens from the attending
        position to the attended-to position.  If bidirectional=False,
        then positive relative positions are invalid.

        We use smaller buckets for small absolute relative_position and
        larger buckets for larger absolute relative_positions. All
        relative positions >=max_distance map to the same bucket.  All
        relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer
        sequences than the model has been trained on.

        Args:
            relative_position: an int32 Tensor

            bidirectional: a boolean - whether the attention is bidirectional

            num_buckets: an integer

            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        relative_buckets = 0
        #        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(relative_position / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = tf.range(query_length)[:, None]
        memory_position = tf.range(key_length)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.position_buckets,
        )
        values = tf.gather(
            self.relative_attention_bias, relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = tf.expand_dims(
            tf.transpose(values, [2, 0, 1]), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        training=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source
        sentence (provided by key_value_states).
        """
        # Input is (batch_size, query_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or
        # (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)

        batch_size, seq_length = shape_list(hidden_states)[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert len(past_key_value) == 2
            real_seq_length += shape_list(past_key_value[0])[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else shape_list(key_value_states)[1]

        def shape(hidden_states):
            """projection"""
            return tf.transpose(
                tf.reshape(hidden_states, (batch_size, -1, self.num_heads, self.head_dim)), perm=(0, 2, 1, 3)
            )

        def unshape(hidden_states):
            """compute context"""
            return tf.reshape(tf.transpose(hidden_states, perm=(0, 2, 1, 3)), (batch_size, -1, self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = tf.concat([past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query
        # (batch_size, n_heads, query_length, dim_per_head)
        query_states = shape(self.q(hidden_states))

        # get key/value
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # to cope with keras serialization
        if self.is_decoder and use_cache:
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None

        # (batch_size, n_heads, query_length, key_length)
        scores = tf.einsum("bnqd,bnkd->bnqk", query_states, key_states)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = tf.zeros((1, self.num_heads, real_seq_length, key_length))
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                position_bias = tf.cast(position_bias, dtype=mask.dtype)
                # (batch_size, n_heads, query_length, key_length)
                position_bias = position_bias + mask

        scores += position_bias
        # (batch_size, n_heads, query_length, key_length)
        weights = tf.nn.softmax(scores, axis=-1)
        # (batch_size, n_heads, query_length, key_length)
        weights = self.dropout(weights, training=training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            tf.debugging.assert_equal(shape_list(layer_head_mask), [self.num_heads])
            weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        # (batch_size, n_heads, query_length, dim_per_head)
        attn_output = tf.matmul(weights, value_states)

        attn_output = self.o(unshape(attn_output))

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (weights,)

        return outputs

    def get_config(self):
        return {
            "dim": self.dim,
            "head_dim": self.head_dim,  # formally kv dim (page 11)
            "inner_dim": self.inner_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            # relative attention
            "position_buckets": self.position_buckets,
            "position_max_distance": self.position_max_distance,
            "position_bidirectional": self.position_bidirectional,
            # others
            "output_attention": self.output_attention,
            "relative_attention_bias": self.relative_attention_bias,
            "has_relative_attention_bias": self.has_relative_attention_bias,
            "use_cache": self.use_cache,
            "is_decoder": self.is_decoder,
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
class T5Block(Layer):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int = None,
        head_dim: int = 64,
        dropout_rate: float = 0.05,
        spatial_dropout_rate: float = 0.0,
        activation: str = "gelu",
        epsilon: float = 1e-6,
        use_gated_ffn: bool = False,
        use_cpe: bool = False,
        position_buckets: int = 32,
        position_max_distance: int = 128,
        position_bidirectional: bool = True,
        output_attention: bool = False,
        has_relative_attention_bias: bool = False,
        **kwargs
    ) -> None:
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
        self.position_buckets = position_buckets
        self.position_max_distance = position_max_distance
        self.position_bidirectional = position_bidirectional
        self.output_attention = output_attention
        self.has_relative_attention_bias = has_relative_attention_bias

        if use_cpe:
            self.cpe = layers.DepthwiseConv1D(kernel_size=1, strides=1)

        self.spatial_drop = layers.SpatialDropout1D(spatial_dropout_rate)

        self.norm1 = T5LayerNorm(epsilon=epsilon)
        self.norm2 = T5LayerNorm(epsilon=epsilon)

        self.attention = T5Attention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            position_buckets=position_buckets,
            position_max_distance=position_max_distance,
            position_bidirectional=position_bidirectional,
            output_attention=output_attention,
            has_relative_attention_bias=has_relative_attention_bias,
        )

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

        if self.use_gated_ffn:
            self.ffn = GatedFFN(
                hidden_size=self.hidden_dim,
                out_size=self.out_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )
        else:
            self.ffn = FFN(
                hidden_size=self.hidden_dim,
                out_size=self.out_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:
        residual = inputs
        x = inputs

        if self.use_cpe:
            x = self.cpe(x)
            x = x + residual

        residual = x

        if self.spatial_dropout_rate:
            x = self.spatial_drop(x, training=training)

        x = self.dropout1(x, training=training)
        attention_outputs = self.attention(x, training=training)
        x = attention_outputs[0]

        if self.output_attention:
            att = attention_outputs[-1]

        x = self.dropout2(x, training=training)

        x = x + residual
        x = self.norm1(x, training=training)

        residual = x
        x = self.ffn(x, training=training)
        x = self.dropout3(x, training=training)

        if self.out_dim == self.dim:
            x = x + residual

        x = self.norm2(x, training=training)

        if self.output_attention:
            return x, att

        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "out_dim": self.out_dim,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "activation": self.activation,
            "epsilon": self.epsilon,
            "use_cpe": self.use_cpe,
            "use_gated_ffn": self.use_gated_ffn,
            "position_buckets": self.position_buckets,
            "position_max_distance": self.position_max_distance,
            "position_bidirectional": self.position_bidirectional,
            "output_attention": self.output_attention,
            "has_relative_attention_bias": self.has_relative_attention_bias,
        }

import tensorflow as tf

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="retvec")
class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_len, vocab_size, embedding_size, truncated_normal_initializer_range=None):
        super(TokenAndPositionEmbedding, self).__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.truncated_normal_initializer_range = truncated_normal_initializer_range

        if truncated_normal_initializer_range:
            embeddings_initializer = tf.keras.initializers.TruncatedNormal(stddev=truncated_normal_initializer_range)
        else:
            embeddings_initializer = "uniform"

        self.token_emb = layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size, embeddings_initializer=embeddings_initializer)
        self.pos_emb = PositionalEmbedding(
            input_shape=(self.max_len, self.embedding_size))

    def call(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        return x

    def get_config(self):
        return {
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'truncated_normal_initializer_range': self.truncated_normal_initializer_range
        }


def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
        min_timescale *
        K.exp(K.arange(num_timescales, dtype=K.floatx()) *
              -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


@tf.keras.utils.register_keras_serializable(package="retvec")
class PositionalEmbedding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


@tf.keras.utils.register_keras_serializable(package="retvec")
class BertPooling(Layer):

    def call(self, x):
        return x[:, 0]

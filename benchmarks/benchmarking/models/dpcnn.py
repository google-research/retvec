from typing import Dict

import tensorflow as tf
from tensorflow import Tensor

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


def from_config(vectorizer: Layer,
                config: Dict) -> tf.keras.Model:
    c = config
    m = config['model']
    v = config['vectorizer']
    d = config['dataset']

    return build_DPCNN_model(vectorizer=vectorizer,
                             vectorizer_type=v['type'],
                             max_len=c['max_len'],
                             num_labels=d['num_labels'],
                             blocks=m.get('blocks', 6),
                             filters=m.get('filters', 256),
                             kernel_size=m.get('kernel_size', 3),
                             pool_size=m.get('pool_size', 3),
                             activation=m.get('activation', 'relu'),
                             final_dropout=m.get('final_dropout', 0),
                             final_activation=m.get(
                                 'final_activation', 'softmax'),
                             pretokenized=v.get('pretokenized', False),
                             embedding_size=v.get('embedding_size'))

def dpcnn_block(x: Tensor,
                activation: str = 'relu',
                filters: int = 256,
                kernel_size: int = 3,
                pool_size: int = 3):
    px = layers.MaxPooling1D(pool_size=pool_size, strides=2, padding='same')(x)
    x = layers.Activation(activation=activation)(px)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=1)(x)
    x = layers.Activation(activation=activation)(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', strides=1)(x)
    x = layers.Add()([x, px])
    return x


def build_DPCNN_model(vectorizer: Layer,
                      vectorizer_type: str,
                      max_len: int,
                      num_labels: int,
                      blocks: int = 6,
                      filters: int = 256,
                      kernel_size: int = 3,
                      pool_size: int = 3,
                      activation: str = 'relu',
                      final_dropout: float = 0.0,
                      final_activation: str = 'softmax',
                      pretokenized: bool = False,
                      embedding_size: int = None) -> tf.keras.Model:
    # fasttext uses a different package to generate the word embeddings
    if vectorizer_type == "fasttext":
        inputs = layers.Input(shape=(max_len, embedding_size))

    else:
        inputs = layers.Input(shape=(max_len,)) if pretokenized else layers.Input(
            shape=(1,), dtype=tf.string)

    if vectorizer:
        embedding = vectorizer(inputs)
    else:
        embedding = inputs

    region_x = layers.Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      strides=1)(embedding)
    x = layers.Activation(activation)(region_x)
    x = layers.Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      strides=1)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      strides=1)(x)

    x = layers.Add()([x, region_x])

    for _ in range(blocks):
        x = dpcnn_block(x, activation, filters, kernel_size, pool_size=3)

    x = layers.Flatten()(x)

    if final_dropout:
        x = layers.Dropout(final_dropout)(x)

    outputs = layers.Dense(num_labels, activation=final_activation)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

from typing import Dict
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


def from_config(vectorizer: Layer,
                config: Dict) -> tf.keras.Model:
    c = config
    m = config['model']
    d = config['dataset']
    v = config['vectorizer']

    return build_RNN_model(vectorizer=vectorizer,
                            vectorizer_type=v['type'],
                            max_len=c['max_len'],
                            num_labels=d['num_labels'],
                            dims=m['dims'],
                            depth=m['depth'],
                            recurrent_layer_type=m['recurrent_layer_type'],
                            dropout=m.get('dropout', 0),
                            final_activation=m.get(
                                'final_activation', 'softmax'),
                            pretokenized=v.get('pretokenized', False),
                            embedding_size=v.get('embedding_size'))


def build_RNN_model(vectorizer: Layer,
                     vectorizer_type: str,
                     max_len: int,
                     num_labels: int,
                     dims: int,
                     depth: int,
                     recurrent_layer_type: str,
                     dropout: float = 0,
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

    x = embedding
    if recurrent_layer_type == 'lstm':
        for _ in range(depth - 1):
            x = layers.Bidirectional(layers.LSTM(
                dims, return_sequences=True, dropout=dropout))(x)
        x = layers.Bidirectional(layers.LSTM(dims))(x)

    elif recurrent_layer_type == 'gru':
        for _ in range(depth - 1):
            x = layers.Bidirectional(layers.GRU(
                dims, return_sequences=True, dropout=dropout))(x)
        x = layers.Bidirectional(layers.GRU(dims))(x)

    else:
        raise ValueError(f"{recurrent_layer_type} is not a valid recurrent_layer_type for RNN.")

    outputs = layers.Dense(num_labels, activation=final_activation)(x)
    return tf.keras.Model(inputs, outputs)

from typing import Dict
import keras_nlp
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from .layers import PositionalEmbedding, BertPooling


def from_config(vectorizer: Layer,
                config: Dict) -> tf.keras.Model:
    c = config
    m = config['model']
    v = config['vectorizer']
    d = config['dataset']

    return build_BERT_model(vectorizer=vectorizer,
                            vectorizer_type=v['type'],
                            max_len=c['max_len'],
                            num_labels=d['num_labels'],
                            hidden_dim=m['hidden_dim'],
                            intermediate_dim=m['intermediate_dim'],
                            encoders=m['encoders'],
                            heads=m['heads'],
                            activation=m['activation'],
                            dropout=m.get('dropout', 0),
                            final_activation=m.get(
                                'final_activation', 'softmax'),
                            pretokenized=v.get('pretokenized', False),
                            embedding_size=v.get('embedding_size'))


def build_BERT_model(vectorizer: Layer,
                     vectorizer_type: str,
                     max_len: int,
                     num_labels: int,
                     hidden_dim: int,
                     intermediate_dim: int,
                     encoders: int,
                     heads: int,
                     activation: str = 'gelu',
                     dropout: float = 0,
                     final_activation: str = 'softmax',
                     pretokenized: bool = False,
                     epsilon: float = 1e-12,
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

    if vectorizer_type not in ['byte_level_bpe']:
        pos_encoding = keras_nlp.layers.PositionEmbedding(sequence_length=max_len)(embedding)
        embedding = embedding + pos_encoding

    x = embedding
    x = layers.LayerNormalization(epsilon=epsilon)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_dim)(x)

    for _ in range(encoders):
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=heads,
            activation=activation,
            layer_norm_epsilon=epsilon,
            dropout=dropout
        )(x)

    x = BertPooling()(x)
    outputs = layers.Dense(num_labels, activation=final_activation)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

from typing import Any, Dict, Tuple
from time import time

from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


def from_config(text: Tensor,
                config: Dict) -> Tuple[Layer, Any]:
    c = config
    v = config['vectorizer']

    vectorizer = get_learned_vectorizer(text=text,
                                        max_len=c['max_len'],
                                        embedding_size=v['embedding_size'],
                                        voc_size=v['voc_size'],
                                        standardize=v['standardize'],
                                        truncated_normal_initializer_range=v.get('truncated_normal_initializer_range', None))

    tokenizer = None
    return vectorizer, tokenizer


def get_learned_vectorizer(text: Tensor,
                                 max_len: int = 128,
                                 embedding_size: int = 200,
                                 voc_size: int = 30000,
                                 truncated_normal_initializer_range: float = None,
                                 standardize='lower') -> Layer:

    vectorizer = layers.TextVectorization(max_tokens=voc_size,
                                          output_sequence_length=max_len,
                                          standardize=standardize,
                                          name='token')
    print('Adaptation Start Learned')
    adapt_start = time()
    vectorizer.adapt(text)
    adapt_time = time() - adapt_start
    print('Adaptation time Learned', adapt_time)

    if truncated_normal_initializer_range:
        embeddings_initializer = tf.keras.initializers.TruncatedNormal(stddev=truncated_normal_initializer_range)
    else:
        embeddings_initializer = "uniform"

    embedding = layers.Embedding(voc_size, embedding_size, embeddings_initializer=embeddings_initializer)
    vectorizer = tf.keras.Sequential([vectorizer, embedding])
    return vectorizer

import codecs
from typing import Any, Dict, Tuple

import numpy as np
from tensorflow import Tensor
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tqdm.auto import tqdm


def from_config(text: Tensor,
                config: Dict) -> Tuple[Layer, Any]:
    c = config
    v = config['vectorizer']

    vectorizer = get_glove_vectorizer(text=text,
                                      glove_path=v['glove_path'],
                                      embedding_size=v['embedding_size'],
                                      max_len=c['max_len'],
                                      voc_size=v['voc_size'],
                                      standardize=v['standardize'])

    # GloVe does not require a separate tokenizer
    tokenizer = None
    return vectorizer, tokenizer


def get_glove_vectorizer(text: Tensor,
                         glove_path: str,
                         embedding_size: int,
                         max_len: int = 128,
                         voc_size: int = 30000,
                         standardize='lower') -> Layer:
    vectorizer = layers.TextVectorization(max_tokens=voc_size,
                                          output_sequence_length=max_len,
                                          standardize=standardize)
    vectorizer.adapt(text)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    num_tokens = len(voc)

    embedding_matrix = build_glove_matrix(glove_path,
                                          embedding_size,
                                          num_tokens,
                                          word_index)
    embedding = layers.Embedding(
        num_tokens,
        embedding_size,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False)

    vectorizer = tf.keras.Sequential([vectorizer, embedding])
    return vectorizer


def build_glove_matrix(glove_path: str,
                       embedding_size: int,
                       num_tokens: int,
                       word_index: Dict) -> np.ndarray:

    # loading glove
    embeddings_index = {}
    with codecs.open(glove_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="reading glove"):
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

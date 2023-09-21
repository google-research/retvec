from typing import Any, Dict, List, Tuple, Union
import os
from time import time

from pathlib import Path
from tqdm.auto import tqdm
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import sentencepiece as spm


def from_config(text: Tensor,
                config: Dict) -> Tuple[Layer, Any]:
    c = config
    v = config['vectorizer']
    d = config['dataset']

    tokenizer = get_sentencepiece_tokenizer(text=text,
                                            model_path=v.get('model_path', None),
                                            voc_size=v.get('voc_size', 30000),
                                            model_type=v.get('model_type', 'unigram'),
                                            lowercase=v.get('lowercase', True),
                                            save_dir=v.get('save_dir', './'),
                                            dataset_name=d['name'],
                                            language=d['language'])

    # Learned embedding does not require a separate tokenizer
    vectorizer = get_sentencepiece_vectorizer(text=text,
                                              embedding_size=v['embedding_size'],
                                              voc_size=v['voc_size'],
                                              truncated_normal_initializer_range=v.get('truncated_normal_initializer_range', None))
    return vectorizer, tokenizer

def get_sentencepiece_tokenizer(text: Tensor,
                                model_path: str = None,
                                voc_size: int = 30000,
                                model_type: str = 'unigram',
                                lowercase: bool = False,
                                save_dir: str = './',
                                dataset_name: str = '',
                                language: Union[str, List[str]] = 'en'):
    default_model_dir = save_dir + 'sentencepiece/models/'
    default_model_prefix = default_model_dir + dataset_name + f"-{voc_size}-{model_type}"
    default_model_path = default_model_prefix + '.model'

    # if model path already defined, we use the model
    if model_path and os.path.isfile(model_path):
        sp = spm.SentencePieceProcessor(model_path)

    elif os.path.isfile(default_model_path):
        sp = spm.SentencePieceProcessor(default_model_path)

    # otherwise we train sentencepiece on the whole dataset
    else:
        language = [language] if isinstance(language, str) else language

        # character coverage set to 0.9995 for japanese and chinese as recommended
        if 'zh' in language or 'ja' in language:
            character_coverage = 0.9995
        else:
            character_coverage = 1.0

        dataset_save_dir = save_dir + 'sentencepiece/datasets/'
        dataset_save_path = dataset_save_dir + dataset_name + '.txt'
        os.makedirs(dataset_save_dir, exist_ok=True)
        os.makedirs(default_model_dir, exist_ok=True)

        text = text.numpy()
        text = [str(s) + '\n' for s in text]

        if lowercase:
            text = [s.lower() for s in text]

        with open(dataset_save_path, 'w') as f:
            for t in text:
                f.write(t)

        print('Adaptation Start Sentencepiece')
        adapt_start = time()
        spm.SentencePieceTrainer.train(input=dataset_save_path,
                                       model_prefix=default_model_prefix,
                                       character_coverage=character_coverage,
                                       vocab_size=voc_size)
        adapt_time = time() - adapt_start
        print('Adaptation time Sentencepiece', adapt_time)

        sp = spm.SentencePieceProcessor(default_model_path)

    return sp

def get_sentencepiece_vectorizer(text: Tensor,
                                 embedding_size: int = 200,
                                 voc_size: int = 30000,
                                 truncated_normal_initializer_range: float = None) -> Layer:
    if truncated_normal_initializer_range:
        embeddings_initializer = tf.keras.initializers.TruncatedNormal(stddev=truncated_normal_initializer_range)
    else:
        embeddings_initializer = "uniform"

    vectorizer = layers.Embedding(voc_size, embedding_size, embeddings_initializer=embeddings_initializer)
    return vectorizer

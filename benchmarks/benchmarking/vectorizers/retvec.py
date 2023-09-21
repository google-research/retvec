
from typing import Any, Dict, Tuple

from tensorflow import Tensor
from tensorflow.keras.layers import Layer

from retvec.tf.layers import RETVecTokenizer


def from_config(text: Tensor,
                config: Dict) -> Tuple[Layer, Any]:
    c = config
    v = config['vectorizer']

    standardize = 'lower' if v.get('lowercase', True) else None
    vectorizer = RETVecTokenizer(
        model=v['model'],
        sequence_length=c['max_len'],
        sep=v.get('sep', ''),
        standardize=standardize,
        trainable=v.get('trainable', False),
        word_length=v.get('max_chars', 16),
        char_encoding_size=v.get('char_encoding_size', 24),
        char_encoding_type=v.get('char_encoding_type', 'UTF-8'),
        dropout_rate=v.get('dropout_rate', 0.0),
        spatial_dropout_rate=v.get('spatial_dropout_rate', 0.0),
        norm_type=v.get('norm_type', None),
        **v.get('kwargs', {}))

    # RetVec does not require a separate tokenizer
    tokenizer = None
    return vectorizer, tokenizer

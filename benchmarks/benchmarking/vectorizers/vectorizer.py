from typing import Any, Dict, Tuple, no_type_check

from tensorflow import Tensor
from tensorflow.keras.layers import Layer

from . import byte_level_bpe, glove, learned, retvec, sentencepiece


VECTORIZER_FNS = {
    'byte_level_bpe': byte_level_bpe.from_config,
    'glove': glove.from_config,
    'learned': learned.from_config,
    'retvec': retvec.from_config,
    'sentencepiece': sentencepiece.from_config
}


@no_type_check
def get_vectorizer(text: Tensor, config: Dict) -> Tuple[Layer, Any]:
    """Get vectorizer layer and tokenizer (if it exists, i.e. huggingface
    Tokenizers) from config and input text. The input is needed
    when tokenizers are adapted or trained on the text, i.e. GloVe.

    Args:
        text (Tensor): [description]
        config (Dict): [description]

    Returns:
        [type]: [description]
    """
    vec_type = config['vectorizer']['type']

    if vec_type not in VECTORIZER_FNS:
        return None, None
    else:
        vec_fn = VECTORIZER_FNS[vec_type]
        return vec_fn(text, config)

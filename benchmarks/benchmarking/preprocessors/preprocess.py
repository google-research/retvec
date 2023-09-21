from typing import Dict, Any

from tensorflow import Tensor
from tensorflow.keras.layers import Layer

from .huggingface import preprocess_huggingface
from .sentencepiece import preprocess_sentencepiece

PREPROCESS_FNS = {
    'byte_level_bpe': preprocess_huggingface,
    'sentencepiece': preprocess_sentencepiece
}


def preprocess_text(text: Tensor,
                    vectorizer: Layer,
                    tokenizer: Any,
                    config: Dict) -> Tensor:
    """Builds the tokenizer layer and applies any preprocessing
    necessary onto the texts, preparing them for training or
    evaluation.

    Args:
        text (Tensor): [description]
        config (Dict): [description]
        tokenizer (Tokenizer): [description]
    """
    pp_fn = PREPROCESS_FNS.get(config['vectorizer']['type'])
    if not pp_fn:
        return text

    return pp_fn(text, vectorizer, tokenizer, config)

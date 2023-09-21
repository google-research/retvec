
from typing import Dict

from tensorflow import Tensor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing


def preprocess_sentencepiece(text: Tensor,
                             vectorizer: Layer,
                             tokenizer: Tokenizer,
                             config: Dict):
    """Preprocessing text for models when using sentencepiece
    before the input text can be fed into the model.

    Args:
        text (Tensor): [description]
        vectorizer (Layer): [description]
        tokenizer (Tokenizer): [description]
        config (Dict): [description]

    Returns:
        [type]: [description]
    """
    tokenized_text = [[tokenizer.bos_id()] + tokenizer.encode_as_ids(str(s)) for s in text.numpy()]
    pad_token = tokenizer.piece_to_id("[PAD]")
    tokenized_text = pad_sequences(
        tokenized_text, maxlen=config['max_len'],
        padding='post', truncating='post', value=pad_token)
    return tokenized_text


from typing import Dict

from tensorflow import Tensor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing


def preprocess_huggingface(text: Tensor,
                           vectorizer: Layer,
                           tokenizer: Tokenizer,
                           config: Dict):
    """Preprocessing text for models when using huggingface
    Tokenizers before the input text can be fed into the model.

    Args:
        text (Tensor): [description]
        vectorizer (Layer): [description]
        tokenizer (Tokenizer): [description]
        config (Dict): [description]

    Returns:
        [type]: [description]
    """
    if config['model']['type'] == 'bert':
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]"))
        )

    tokenizer.enable_truncation(max_length=config['max_len'])
    tokenized_text = [tokenizer.encode(str(s)).ids for s in text.numpy()]

    pad_token = tokenizer.token_to_id("[PAD]")
    tokenized_text = pad_sequences(
        tokenized_text, maxlen=config['max_len'],
        padding='post', truncating='post', value=pad_token)
    return tokenized_text

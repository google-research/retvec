
import codecs
import os
import tempfile
from time import time
from typing import Any, Dict, Tuple, Optional

from tensorflow import Tensor
from tokenizers import ByteLevelBPETokenizer
from tensorflow.keras.layers import Layer

from ..models.layers import TokenAndPositionEmbedding


SPECIAL_TOKENS = [
    "[CLS]",
    "[SEP]",
    "[UNK]",
    "[PAD]",
    "[MASK]"
]


def from_config(text: Tensor,
                config: Dict) -> Tuple[Layer, Any]:
    c = config
    v = config['vectorizer']
    vectorizer = TokenAndPositionEmbedding(max_len=c['max_len'],
                                           vocab_size=v['voc_size'],
                                           embedding_size=v['embedding_size'])

    tokenizer = get_byte_level_bpe_tokenizer(text=text,
                                             save_dir=c['base_dir'],
                                             exp_name=c['exp_name'],
                                             voc_size=v['voc_size'],
                                             lowercase=v['lowercase'],
                                             min_frequency=v.get(
                                                 'min_frequency', 2))
    return vectorizer, tokenizer


def get_byte_level_bpe_tokenizer(text: Tensor,
                                 save_dir: str,
                                 exp_name: str,
                                 voc_size: int,
                                 lowercase: bool = True,
                                 min_frequency: Optional[int] = 2):
    """[summary]

    Args:
        text (Tensor): [description]
        save_dir (Optional[str], optional): [description]. Defaults to None.
        voc_size (Optional[int], optional): [description]. Defaults to None.
        min_frequency (Optional[int], optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    # load tokenizer from pretrained vocab and merges file
    save_dir = save_dir + 'bpe_tokenizers/' + exp_name
    vocab_file = save_dir + '/vocab.json'
    merges_file = save_dir + '/merges.txt'

    # if tokenizer exists load it, otherwise train and save a new tokenizer
    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        tokenizer = ByteLevelBPETokenizer(vocab_file,
                                          merges_file,
                                          lowercase=lowercase)
        return tokenizer
    else:
        # Dump dataset into a tmp file because huggingface tokenizers requires
        _, path = tempfile.mkstemp()
        try:
            with codecs.open(path, 'w', 'utf-8') as f:
                for t in text.numpy():
                    f.write(str(t) + '\n')

            # initialize and train tokenizer on text
            # TODO (marinazh); WordPiece and Unigram
            tokenizer = ByteLevelBPETokenizer()

            print('Adaptation Start BPE')
            adapt_start = time()
            tokenizer.train(files=[path],
                            vocab_size=voc_size,
                            min_frequency=min_frequency,
                            special_tokens=SPECIAL_TOKENS)
            adapt_time = time() - adapt_start
            print('Adaptation time BPE', adapt_time)
        finally:
            os.remove(path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        tokenizer.save_model(save_dir)
        return tokenizer

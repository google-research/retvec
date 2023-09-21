
from functools import partial
import random
from typing import Any, List, Sequence, Tuple, Union
from multiprocess import Pool
from nltk.tokenize import word_tokenize


import numpy as np

import editdistance

from .ops import (
    insert,
    delete,
    swap,
    substitute,
    typo
)

from .data.alphabets import (CHINESE_CHARACTERS, ENGLISH_CHARACTERS,
                             FRENCH_CHARACTERS, GERMAN_CHARCTERS,
                             JAPANESE_CHARACTERS, SPANISH_CHARACTERS,
                             URDU_CHARACTERS, KOREAN_CHARACTERS)


ALPHABETS = {
    'en': ENGLISH_CHARACTERS,
    'fr': FRENCH_CHARACTERS,
    'de': GERMAN_CHARCTERS,
    'es': SPANISH_CHARACTERS,
    'zh': CHINESE_CHARACTERS,
    'ja': JAPANESE_CHARACTERS,
    'ur': URDU_CHARACTERS,
    'ko': KOREAN_CHARACTERS,
}

OPS = {
    'insert': insert,
    'swap': swap,
    'substitute': substitute,
    'delete': delete,
    'typo': typo
}


class CharAugmenter():

    def __init__(self,
                 language: Union[str, List[str]],
                 ops: List[str],
                 use_multiprocess: bool = False,
                 min_len: int = 4) -> None:
        self.ops = ops
        self.language = language

        # get alphabet based on language, combine alphabets
        # for lists of languages
        if isinstance(self.language, str):
            self.alphabet = ALPHABETS.get(self.language, ENGLISH_CHARACTERS)
        else:
            alphabet = []
            for lang in self.language:
                alphabet.extend(ALPHABETS.get(lang, []))
            self.alphabet = alphabet

        self.use_multiprocess = use_multiprocess
        self.min_len = min_len

        self.op_fns = []
        for op_name in ops:
            op = OPS[op_name]

            # insert and substitution based on the corresponding alphabet
            if op_name in ['insert', 'substitute']:
                op = partial(op, dataset=self.alphabet)
            self.op_fns.append(op)

    def __call__(self,
                 texts: Sequence[str],
                 labels: Sequence[Any],
                 block_size: int = 1,
                 percent: float = 0.1,
                 augmentations_per_example: int = 1) -> Tuple[List, List]:
        """[summary]

        Args:
            text: [description]
            labels: [description]
            block_size: [description].
            pct_words_to_swap: [description].
            augmentations_per_example: [description].

        Returns:
            List[Tuple[str, int]]: [description]
        """
        # wrapper for multiprocessing
        def _multi_run_wrapper(idx):
            text = texts[idx]
            label = labels[idx]
            return self._augment(text, label, block_size, percent,
                                 augmentations_per_example)

        output_texts = []
        output_labels = []

        # FIXME: use multiprocessing?
        for i in range(len(texts)):
            aug_texts, aug_labels = self._augment(texts[i], labels[i],
                                                  block_size, percent,
                                                  augmentations_per_example)
            output_texts.extend(aug_texts)
            output_labels.extend(aug_labels)

        return output_texts, output_labels

    def _augment(self,
                 text: str,
                 label: Any,
                 block_size: int = 1,
                 pct_words_to_swap: float = 0.1,
                 augmentations_per_example: int = 1) -> Tuple[List, List]:
        """Helper function that augments a single input text."""
        output_texts = []
        output_labels = []

        if isinstance(text, bytes):
            text = text.decode('utf-8')

        if not pct_words_to_swap:
            output_texts = [text] * augmentations_per_example
            output_labels = [label] * augmentations_per_example
            return output_texts, output_labels

        for _ in range(augmentations_per_example):

            # reset for new augmentation
            if self.language in ['zh', 'ja']:
                cnt = 0
                op_matrix = np.zeros((len(text), len(self.op_fns)))
                aug_text = text

                # collapse all whitespace because we want to insert it later
                num_words_to_swap = int(max(len(aug_text) * pct_words_to_swap, 1))

                # selecting an upper bound of attempts
                while cnt < num_words_to_swap:

                    # selecting op
                    op_id = random.randint(0, len(self.op_fns) - 1)
                    op = self.op_fns[op_id]

                    bsize = random.randint(1, block_size)

                    # applying augmentation op to text
                    aug_text, op_matrix = op(
                        aug_text, op_matrix, op_id, block_size=bsize)

                    # increment counter
                    cnt += 1

            else:
                # other langauges split on whitespace
                # words = text.split()

                words = word_tokenize(text)

                # select pct_words_to_swap words to modify
                idxs_to_modify = self._get_indices_to_modify(
                    words, self.min_len)

                # num_words_to_swap = int(max(len(idxs_to_modify) * pct_words_to_swap, 1))

                random.shuffle(idxs_to_modify)
                idxs_to_modify = idxs_to_modify
                
                # modify selected words using random augmentations once
                for i in idxs_to_modify:
                    if random.random() < pct_words_to_swap:
                        word = str(words[i])
                        op_matrix = np.zeros((len(word), len(self.op_fns)))
                        op_id = random.randint(0, len(self.op_fns) - 1)
                        op = self.op_fns[op_id]

                        bsize = random.randint(1, block_size)

                        aug_word, _ = op(word, op_matrix, op_id,
                                         block_size=bsize)
                        words[i] = aug_word

                aug_text = ' '.join(words)

            output_texts.append(aug_text)
            output_labels.append(label)

        return output_texts, output_labels

    def _get_indices_to_modify(self,
                               words: Sequence[str],
                               min_len: int = 4) -> List[int]:
        """[summary]

        Args:
            words (Sequence[str]): [description]
            min_len (int, optional): [description]. Defaults to 4.
        """
        idxs = []
        for i in range(len(words)):
            if len(words[i]) < min_len:
                continue
            idxs.append(i)
        return idxs


class CharInsertionAugmenter(CharAugmenter):

    def __init__(self, **kwargs) -> None:
        ops = ['insert']
        super().__init__(ops=ops, **kwargs)


class CharDeletionAugmenter(CharAugmenter):

    def __init__(self, **kwargs) -> None:
        ops = ['delete']
        super().__init__(ops=ops, **kwargs)


class CharSwapAugmenter(CharAugmenter):

    def __init__(self, **kwargs) -> None:
        ops = ['swap']
        super().__init__(ops=ops, **kwargs)


class CharSubstituteAugmenter(CharAugmenter):

    def __init__(self, **kwargs) -> None:
        ops = ['substitute']
        super().__init__(ops=ops, **kwargs)


class CharTypoAugmenter(CharAugmenter):

    def __init__(self, **kwargs) -> None:
        ops = ['typo']
        super().__init__(ops=ops, **kwargs)


class CharMixedAugmenter(CharAugmenter):

    def __init__(self, **kwargs) -> None:
        ops = ['insert', 'delete', 'swap', 'substitute', 'typo']
        super().__init__(ops=ops, **kwargs)


AUG_TYPE_TO_AUGMENTER = {
    'insert': CharInsertionAugmenter,
    'delete': CharDeletionAugmenter,
    'swap': CharSwapAugmenter,
    'substitute': CharSubstituteAugmenter,
    'typo': CharTypoAugmenter,
    'mixed': CharMixedAugmenter
}

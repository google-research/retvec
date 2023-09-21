
import random
from typing import Dict, List, Sequence, Union, Tuple

import numpy as np

from .data.qwerty_typos import QWERTY_TYPOS
from .utils import (get_block_size, get_modified_idxs,
                    get_random_available_block, replace_block,
                    update_op_matrix)


def insert(text: str,
           op_matrix: np.ndarray,
           op_id: int,
           block_size: int,
           dataset: Sequence[str]) -> Tuple[str, np.ndarray]:
    """Insert a block of characters."""
    block_size = get_block_size(text, block_size)
    block_idxs = get_random_available_block(text, op_matrix, block_size)
    if block_idxs:
        start_idx, _ = block_idxs
    else:
        return text, op_matrix

    end_idx = start_idx + block_size
    char = random.choice(dataset)

    # insert block
    block = char * block_size
    aug_s = text[:start_idx] + block + text[start_idx:]

    modified_idxs = range(start_idx, end_idx)
    op_matrix = update_op_matrix(
        op_matrix, modified_idxs, op_id, op_type='insert')
    return aug_s, op_matrix


def swap(text: str,
         op_matrix: np.ndarray,
         op_id: int,
         block_size: int,) -> Tuple[str, np.ndarray]:
    """Swap two contiguous blocks of characters."""
    block_size = get_block_size(text, 2 * block_size)
    block_size = block_size // 2

    # get block indices
    block_idxs = get_random_available_block(
        text, op_matrix, 2*block_size)
    if block_idxs:
        start_idx, end_idx = block_idxs
    else:
        return text, op_matrix

    # build blocks
    mid_idx = start_idx + block_size
    block1 = text[start_idx:mid_idx]
    block2 = text[mid_idx:end_idx]

    # swap
    aug_s = replace_block(text, block2, start_idx)
    aug_s = replace_block(aug_s, block1, mid_idx)

    idxs = range(start_idx, end_idx)
    modified_idxs = get_modified_idxs(text, aug_s, idxs)
    op_matrix = update_op_matrix(op_matrix, modified_idxs, op_id)
    return aug_s, op_matrix


def delete(text: str,
           op_matrix: np.ndarray,
           op_id: int,
           block_size: int) -> Tuple[str, np.ndarray]:
    """Delete a block of characters."""
    if len(text) < 4:
        return text

    block_size = get_block_size(text, block_size)
    block_idxs = get_random_available_block(
        text, op_matrix, block_size)

    if block_idxs:
        start_idx, end_idx = block_idxs
    else:
        return text, op_matrix

    # delete
    aug_s = text[:start_idx] + text[end_idx:]

    modified_idxs = range(start_idx, end_idx)
    op_matrix = update_op_matrix(
        op_matrix, modified_idxs, op_id, op_type='delete')
    return aug_s, op_matrix


def substitute(text: str,
               op_matrix: np.ndarray,
               op_id: int,
               block_size: int,
               dataset: Union[Dict, List]) -> Tuple[str, np.ndarray]:
    """Substitute a block of characters, based on a `dataset`
    that is either a list of replacements for any character
    or a dataset of potential replacements (i.e. {'o': [0, O], ...})
    for designated characters.
    """
    block_size = get_block_size(text, block_size)
    block_idxs = get_random_available_block(
        text, op_matrix, block_size)

    if block_idxs:
        start_idx, end_idx = block_idxs
    else:
        return text, op_matrix

    modified_idxs = []
    aug_s = text
    for i in range(start_idx, end_idx):
        cur = text[i]

        # get replacement in dataset, if available
        # otherwise do not subsitute
        if isinstance(dataset, dict):
            confusable = dataset.get(cur, [cur])
            replacement = random.choice(confusable)
        else:
            replacement = random.choice(dataset)

        # keep track of modified indices for matrix
        if replacement != cur:
            modified_idxs.append(i)
        aug_s = replace_block(aug_s, replacement, i)

    op_matrix = update_op_matrix(op_matrix, modified_idxs, op_id)
    return aug_s, op_matrix


def typo(text: str,
         op_matrix: np.ndarray,
         op_id: int,
         block_size: int,) -> Tuple[str, np.ndarray]:
    """Substitute a block of characters based on QWERTY
    keyboard typos."""
    return substitute(text, op_matrix, op_id, block_size, QWERTY_TYPOS)

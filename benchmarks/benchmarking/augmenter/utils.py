import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Unmodified (identity) op id for op matrices
UNMODIFIED = 0

# Modified op id for op matrices
MODIFIED = 1


def get_block_size(text: str,
                   block_size: int) -> int:
    return min(block_size, len(text))


def replace_block(s: str,
                  new_block: str,
                  start_idx: int) -> str:
    """Replace a block of characters in a string."""
    block_len = len(new_block)
    left = s[:start_idx]
    right = s[start_idx + block_len:]
    return left + new_block + right


def get_modified_idxs(original_text: str,
                      aug_text: str,
                      indices: Sequence[int]) -> Sequence[int]:
    """Get modified indices for substitution-based operations,
    in case the op fails to modify the string.

    Length-modifying ops like insertion/deletion will be computed
    separately because they are always modified."""
    modified_idxs = []
    for i in indices:
        if original_text[i] != aug_text[i]:
            modified_idxs.append(i)
    return modified_idxs


def update_op_matrix(op_matrix: np.ndarray,
                     indices: Sequence[int],
                     op_id: int,
                     op_type: Optional[str] = None) -> np.ndarray:

    # if insertion, we need to shift the rows after the insertion
    # point downwards to adjust with the shift caused by the insert

    if op_type == "insert":
        insert_row = np.zeros(shape=(op_matrix.shape[1]))
        insert_row[op_id] = MODIFIED

        for i in indices:
            op_matrix = np.insert(op_matrix, i, insert_row, axis=0)

        # keep op_matrix the same shape
        op_matrix = op_matrix[:-len(indices)][:]

    # FIXME: is this assumption reasonable?
    # if deletion, we shift the rows upwards to adjust. This means that
    # for some rows, it's possible to have multiple augmentation
    # ops marked in the op_matrix (deletion and another op).
    elif op_type == "delete":

        # remove rows corresponding to deleted elements in matrix
        op_matrix = np.delete(op_matrix, indices, axis=0)

        # keep matrix the size with extra zeros
        zeros = np.zeros(shape=(len(indices), op_matrix.shape[1]))
        op_matrix = np.concatenate([op_matrix, zeros], axis=0)

        # keep track of deletion op in matrix
        for i in indices:
            op_matrix[i][op_id] = MODIFIED

    # general case for ops that do not add/remove elements
    else:
        for i in indices:
            # check that we have not modified the index before
            # otherwise, something went wrong
            assert op_matrix[i][op_id] == UNMODIFIED
            op_matrix[i][op_id] = MODIFIED

    return op_matrix


def get_random_available_block(s: str,
                               op_matrix: np.array,
                               block_size: int = 1) -> Optional[Tuple[int, int]]:  # noqa
    """Returns a random available block with size `block_size`
       or None if there are no available blocks.
    """
    blocks = get_all_available_blocks(s, op_matrix, block_size)
    if len(blocks) and len(blocks.get(block_size, [])):
        return random.choice(blocks[block_size])
    return None


def get_all_available_blocks(s: str,
                             op_matrix: np.array,
                             max_block_size: int = 1) -> Dict[int, List[Tuple[int, int]]]:  # noqa
    """Finds all available blocks for block sizes between 1
    and ``max_block_size`, with the condition that no block intersects
    any previously augmented parts of string `s` as denoted in `aug_matrix`.

    Args:
        s: Current string.
        op_matrix: Matrix tracking augmentations for string `s`,
            where element (i, j) is the op number for char i and op j.
        max_block_size: Maximum length of blocks to query for. Defaults to 1.

    Returns:
        Dictionary mapping block size (all values between 1 and
        `max_block_size`) to a list of available blocks of that size,
        each represented by (start_idx (inclusive), end_idx (exclusive)).
    """
    op_rows = np.sum(op_matrix, axis=1)
    avail_pos = [i for i, x in enumerate(op_rows) if x == UNMODIFIED]

    # dict mapping block size to [start_idx, end_idx] of
    # the list of all available blocks with block size
    avail_blocks = defaultdict(list)
    avail_blocks[1] = [(pos, pos+1) for pos in avail_pos if pos < len(s)]

    for idx, pos in enumerate(avail_pos):
        for size in range(1, max_block_size):
            # prevent out of bounds
            if idx + size >= len(avail_pos) or pos + size >= len(s):
                break

            if avail_pos[idx + size] == pos + size:
                avail_blocks[size + 1].append((pos, pos + size + 1))
            else:
                break

    return avail_blocks

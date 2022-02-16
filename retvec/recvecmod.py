"""
 Copyright 2021 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from .types import FloatTensor, Tensor


@tf.keras.utils.register_keras_serializable(package="retvec")
class RecVecMod(tf.keras.layers.Layer):
    "RecVec: Resilient and Efficient Characters Vectorizer"

    def __init__(self,
                 max_len: int = 16,
                 embedding_size: int = 32,
                 lower_case: bool = False,
                 folds: int = 0,
                 masks: int = 0,
                 primes: List = None,
                 positional_encoding: str = None,
                 cls_int: int = None,
                 replacement_char: int = 11,
                 is_eager: bool = False,
                 clipping_method: str = "binary",
                 **kwargs) -> None:
        """Build a characters to tensor embedding.

        Args:
            max_len: Maximum number of characters to embeded. When adding a
            CLS token (for tranformers based word models). The effective size
            would be max_len + 1.

            embedding_size: [description]

            lower_case: [description]. Defaults to False.

            folds: [description]. Defaults to 0.

            masks: How many chars to mask. Defaults to 0.

            primes: List of primes to use. If None, use the default one.
            Optional, defaults to None which are the provided list.

            positional_encoding: Which type of positional encoding to add,
            if any. Default to int which add the position as int. None disable.

            cls_char: prepend a CLS token. Defaults to None which disable it.

            eager: is the layer running as standalone in eager mode?

            cpu_onehot: use cpu to perform one_hot encoding? Usually way faster
            than GPU

            clipping_method: who to clip the value of the embeddings? binary
            means <1 = 0 >1 ==1. scaling means val / primes.

        Raises:
            ValueError: [description]
        """
        super(RecVecMod, self).__init__()

        # users parasms
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.lower_case = lower_case
        self.folds = folds
        self.replacement_char = replacement_char
        self.masks = masks
        self.is_eager = is_eager
        self.positional_encoding = positional_encoding
        self.clipping_method = clipping_method

        if embedding_size % 2:
            raise ValueError("Embedding size must be a multiple of 2")

        if folds % 2:
            raise ValueError("Fold must be a multiple of 2")

        # convert to tf.constant
        self.tf_embedding_size = tf.constant(self.embedding_size)
        self.tf_max_len = tf.constant(self.max_len)
        self.tf_is_eager = tf.constant(self.is_eager)

        PRIMES = {
            "24": [23, 43, 103, 73, 71, 131, 101, 37, 29, 2243, 149, 569, 97, 107, 19, 1493],  # noqa 16
            "32": [19, 23, 31, 37, 67, 83, 107, 109, 131, 139, 149, 283, 1109],  # noqa 13
            "40": [17, 23, 31, 37, 53, 59, 67, 71, 83, 89, 499],  # 11
            "48": [31, 43, 59, 61, 89, 199, 941, 66343]  # 8
        }

        # primes for encoding
        self.primes = primes  # keep for serialization
        if primes is not None:
            self.primes_list = tf.constant(self.primes, dtype='int32')
        else:
            k = "%s" % embedding_size
            if k not in PRIMES:
                raise ValueError("Only support embedding of size: 24,32,40,48")
            primes_list = PRIMES[k]
            self.primes_list = tf.constant(primes_list, dtype='int32')

        # [CLS] character injection
        self.cls_int = cls_int  # needed for serialization
        if self.cls_int:
            self.pad_position = tf.constant([[0, 0], [1, 0]])
            self.pad_value = tf.constant(cls_int)
            self.output_size = max_len + 1
            print("Output size extended by 1 to inject CLS token")
        else:
            self.output_size = max_len

        # masking
        # building mask of shape str_len, embedding_size with n "step" at 1..1
        mask = np.zeros((self.output_size, self.embedding_size), dtype='int32')
        for idx in range(self.masks):
            # note we will shuffle so the mask position doesn't matter
            # replace with 11..1 a given step
            mask[idx] = np.ones(self.embedding_size, dtype='int32')
        self.mask = tf.constant(mask, dtype='int32')

        # position vectors ()
        # folding position vector inside the fold != step positional encoding.
        if folds:
            self.fold_size = max_len // folds
            # generate within fold positional vector
            # shape: ([1, 2, 3, 4, ... 1, 2, 3, 4....]
            fold_vect = tf.constant([i for i in range(folds)], dtype='int32')
            self.fold_position_vector = tf.tile(fold_vect, [self.fold_size])

            # recompute the output size which is now [CLS] + Folds at most
            self.output_size = self.fold_size
            self.output_size += 1 if self.cls_int else 0  # add 1 for CLS?
            self.position_vector_size = self.fold_size
        else:
            self.position_vector_size = max_len

        # positional encoding vector generation
        position_arr = [i for i in range(self.position_vector_size)]
        self.position_vector = tf.constant(position_arr, dtype='int32')

    @tf.function()
    def call(self, inputs: Tensor) -> FloatTensor:

        if self.lower_case:
            inputs = tf.strings.lower(inputs)

        # print('inputs', inputs.shape)
        chars = tf.strings.unicode_decode(
            inputs,
            'UTF-8',
            errors='replace',
            replacement_char=self.replacement_char)

        # print('chars', chars.shape)
        # chars = tf.reshape(chars, (chars.shape[0], chars.shape[1]))
        # encode
        if self.is_eager:
            chars = chars.to_tensor(shape=(chars.shape[0], self.tf_max_len))
        else:
            chars = chars.to_tensor(shape=(chars.shape[0], 1, self.tf_max_len))
            # print('to_tensor', chars.shape)
            # needed in graph mode
            chars = tf.squeeze(chars, axis=1)

        # perform folding if requested
        if self.folds:
            # add positional
            chars = tf.add(chars, self.fold_position_vector)

            # reshape
            fold_shape = [tf.shape(chars)[0], self.fold_size, self.folds]
            chars = tf.reshape(chars, fold_shape)

            # sum the fold values
            chars = tf.reduce_sum(chars, axis=-1)

        # apply position vector (after folding!)
        if self.positional_encoding:
            chars = tf.add(chars, self.position_vector)

        # # add CLS if needed
        if self.cls_int:
            chars = tf.pad(chars,
                           self.pad_position,
                           constant_values=self.pad_value)

        # project into smaller space
        embeddings = self._project(chars, self.primes_list,
                                   self.tf_embedding_size)

        # fast masking if needed and layer trainable
        if self.trainable and self.masks:
            # randomize mask positions
            mask = tf.random.shuffle(self.mask)
            # add them
            embeddings = embeddings + mask

        # !clipping must be the last op before casting.
        # clip values as there are collision that push the val above 1
        embeddings = tf.clip_by_value(embeddings, 0, 1)

        # cast to float to be like a normal layer output
        final_embeddings: FloatTensor = tf.cast(embeddings, dtype='float32')

        return final_embeddings

    @tf.function(experimental_compile=True)
    def _project(self, chars: FloatTensor, primes_list: List[int],
                 embedding_size: int) -> FloatTensor:
        """Project chars in subspace"""
        # intial tensor
        embeddings = tf.math.mod(chars, primes_list[0])
        embeddings = tf.one_hot(embeddings, embedding_size, dtype='int32')

        # subsequent space projection are added
        # using tf_map() seems to make it slower experimentally
        for idx in range(1, len(primes_list)):

            # prime
            dim = tf.math.mod(chars, primes_list[idx])

            # overflow prevention when prime > embedding size
            # Or do w need to clip instead to reduce collision TBT
            dim = tf.math.mod(dim, embedding_size)
            dim = tf.one_hot(dim, embedding_size, dtype='int32')
            embeddings = tf.add(embeddings, dim)
        return embeddings

    def get_config(self) -> Dict[str, Any]:
        return {
            'max_len': self.max_len,
            'embedding_size': self.embedding_size,
            'lower_case': self.lower_case,
            'folds': self.folds,
            'masks': self.masks,
            "primes": self.primes,
            'replacement_char': self.replacement_char,
            "positional_encoding": self.positional_encoding,
            "cls_int": self.cls_int,
            "is_eager": self.is_eager,
            "clipping_method": self.clipping_method
        }

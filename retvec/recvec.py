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

from typing import Any, Dict

import numpy as np
import tensorflow as tf

from .types import FloatTensor, Tensor


@tf.keras.utils.register_keras_serializable(package="retvec")
class RecVec(tf.keras.layers.Layer):
    "RecVec: Resilient and Efficient Characters Vectorizer"

    def __init__(self,
                 max_len: int = 16,
                 embedding_size: int = 16,
                 lower_case: bool = False,
                 folds: int = 0,
                 masks: int = 0,
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

            embedding_size: Size of output character embedding.

            lower_case: Whether to lowercase inputs before embedding.

            folds: How many folds to use. 0 to disable.

            masks: How many chars to mask. 0 to disable.

            cls_char: prepend a CLS token. Defaults to None which disable it.

            eager: is the layer running as standalone in eager mode?

            clipping_method: who to clip the value of the embeddings? binary
            means <1 = 0 >1 ==1. scaling means val / primes.
        """
        super(RecVec, self).__init__()

        # users parasms
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.lower_case = lower_case
        self.folds = folds
        self.replacement_char = replacement_char
        self.masks = masks

        self.bits_masks = tf.bitwise.left_shift(tf.ones([], dtype='int32'),
                                                tf.range(embedding_size,
                                                         dtype='int32'))
        self.is_eager = is_eager
        self.clipping_method = clipping_method

        if embedding_size % 2:
            raise ValueError("Embedding size must be a multiple of 2")

        if folds % 2:
            raise ValueError("Fold must be a multiple of 2")

        # convert to tf.constant
        self.tf_embedding_size = tf.constant(self.embedding_size)
        self.tf_max_len = tf.constant(self.max_len)
        self.tf_is_eager = tf.constant(self.is_eager)

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

    @tf.function()
    def call(self, inputs: Tensor) -> FloatTensor:
        inputs = tf.stop_gradient(inputs)

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

            # reshape
            fold_shape = [tf.shape(chars)[0], self.fold_size, self.folds]
            chars = tf.reshape(chars, fold_shape)

            # sum the fold values
            chars = tf.reduce_sum(chars, axis=-1)

        # # add CLS if needed
        if self.cls_int:
            chars = tf.pad(chars,
                           self.pad_position,
                           constant_values=self.pad_value)

        # project into smaller space
        embeddings = self._project(chars, self.bits_masks)

        # fast masking if needed and layer trainable
        if self.trainable and self.masks:
            # randomize mask positions
            mask = tf.random.shuffle(self.mask)
            # add them
            embeddings = embeddings + mask

        # cast to float to be like a normal layer output
        final_embeddings: FloatTensor = tf.cast(embeddings, dtype='float32')

        return final_embeddings

    @tf.function(experimental_compile=True)
    def _project(self, chars: FloatTensor, masks: FloatTensor) -> FloatTensor:
        """Project chars in subspace"""
        masked = tf.bitwise.bitwise_and(tf.expand_dims(chars, -1), masks)
        out = tf.cast(tf.not_equal(masked, 0), 'int32')
        return out

    def get_config(self) -> Dict[str, Any]:
        return {
            'max_len': self.max_len,
            'embedding_size': self.embedding_size,
            'lower_case': self.lower_case,
            'folds': self.folds,
            'masks': self.masks,
            'replacement_char': self.replacement_char,
            "cls_int": self.cls_int,
            "is_eager": self.is_eager,
            "clipping_method": self.clipping_method
        }

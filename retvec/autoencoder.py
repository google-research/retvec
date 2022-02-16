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

import random

import tensorflow as tf
from tabulate import tabulate

from retvec.types import FloatTensor, TensorLike


def decode(decoded: TensorLike, input_str: str, verbose: int = 1) -> str:
    "Display the decoded value of the output"
    decoded_ints = tf.argmax(decoded[0], axis=-1)
    decoded_bytes = tf.strings.unicode_encode(
        tf.cast(decoded_ints, dtype='int32'), 'UTF-8')

    decoded_str = str(decoded_bytes.numpy())
    decoded_str = decoded_str[2:-1].replace('\\x00', '')
    if verbose:
        rows = [['input', len(input_str), input_str],
                ['decoded_bytes', '', str(decoded_bytes)[12:len(input_str)*2]],
                ['decoded', len(decoded_str), str(decoded_str)]]
        print(tabulate(rows, headers=['len', 'value']))
    return decoded_str


def autoencode(x: TensorLike, string_len: int,
               encode_size: int) -> FloatTensor:
    # ! keep it as a separated function

    # print('inputs', inputs.shape)
    chars = tf.strings.unicode_decode(x,
                                      'UTF-8',
                                      errors='replace',
                                      replacement_char=0)

    # print('chars', chars.shape)
    # chars = tf.reshape(chars, (chars.shape[0], chars.shape[1]))
    # encode
    chars = chars.to_tensor(shape=(chars.shape[0], string_len))
    chars = tf.math.mod(chars, encode_size)
    # one_hot
    one_hot: FloatTensor = tf.one_hot(chars, encode_size, dtype='float32')
    return one_hot


class AutoEncodeGenerator(tf.keras.utils.Sequence):
    """Generates random word data, for training baseline REW* models."""

    def __init__(self, charset: list, string_len: int, num_batch: int,
                 batch_size: int, encode_size: int, lower_case: bool) -> None:
        self.charset = charset
        self.num_batch = num_batch
        self.batch_size = batch_size
        self.string_len = string_len
        self.encode_size = encode_size
        self.lower_case = lower_case

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index: int) -> TensorLike:

        # let's be blunt about this
        x = []
        for _ in range(self.batch_size):
            s = random.sample(self.charset, self.string_len)
            x.append("".join(s))

        y = autoencode(x=x,
                       string_len=self.string_len,
                       encode_size=self.encode_size)

        x = tf.ragged.constant(x, dtype='string')
        return x, y

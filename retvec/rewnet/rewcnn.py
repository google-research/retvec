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

from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers

from retvec import RetVecBinarizer

from .layers import ConvNextBlock
from .outputs import build_outputs


def build_rewcnn_from_config(config: Dict) -> tf.keras.Model:
    m = config["model"]
    o = config["outputs"]
    return REWCNN(
        max_chars=m["max_chars"],
        char_encoding_size=m["char_encoding_size"],
        char_encoding_type=m["char_encoding_type"],
        replacement_int=m["replacement_int"],
        encoder_blocks=m["encoder_blocks"],
        encoder_hidden_dim=m["encoder_hidden_dim"],
        encoder_kernel_sizes=m["encoder_kernel_sizes"],
        encoder_depth_multiplier=m["encoder_depth_multiplier"],
        encoder_filters=m["encoder_filters"],
        encoder_dropout=m["encoder_dropout"],
        encoder_epsilon=m["encoder_epsilon"],
        encoder_activation=m["encoder_activation"],
        encoder_strides=m["encoder_strides"],
        encoder_norm_type=m["encoder_norm_type"],
        encoder_output_dim=m["encoder_output_dim"],
        encoder_output_activation=m["encoder_output_activation"],
        tokenizer_dense_dim=m["tokenizer_dense_dim"],
        tokenizer_activation=m["tokenizer_activation"],
        similarity_dim=o["similarity_dim"],
        original_decoder_size=o["original_decoder_size"],
        aug_decoder_size=o["aug_decoder_size"],
        aug_vector_dim=o["aug_vector_dim"],
        aug_matrix_dim=o["aug_matrix_dim"],
        outputs_dropout_rate=o["outputs_dropout_rate"],
        outputs_norm_type=o["outputs_norm_type"],
        similarity_norm_type=o["similarity_norm_type"],
    )


@tf.keras.utils.register_keras_serializable(package="retvec")
def REWCNN(
    max_chars: int = 16,
    char_encoding_size: int = 32,
    char_encoding_type: str = "UTF-8",
    replacement_int: int = 11,
    encoder_blocks: int = 2,
    encoder_hidden_dim: int = 32,
    encoder_kernel_sizes: List[int] = [5, 5],
    encoder_depth_multiplier: int = 2,
    encoder_filters: int = 32,
    encoder_dropout: float = 0.0,
    encoder_epsilon: float = 1e-10,
    encoder_activation: str = "relu",
    encoder_strides: List[int] = [1, 1],
    encoder_norm_type: str = "batch",
    encoder_output_dim: int = 0,
    encoder_output_activation: str = None,
    tokenizer_dense_dim: int = 0,
    tokenizer_activation: str = "tanh",
    similarity_dim: int = 128,
    original_decoder_size: int = 0,
    aug_decoder_size: int = 0,
    aug_vector_dim: int = 0,
    aug_matrix_dim: int = 0,
    outputs_dropout_rate: float = 0.0,
    outputs_norm_type: str = None,
    similarity_norm_type: str = "l2",
) -> tf.keras.Model:
    """REWCNN model based on ConvNet architecture.

    The model is based on the ConvNet architecture, introduced in the paper
    A ConvNet for the 2020s (https://arxiv.org/abs/2201.03545).

    Args:
        max_chars: Maximum number of characters to binarize. If the input
            is 2d, i.e. (batch_size, num_words) this is still the max
            characters per word.

        char_encoding_size: Size of output character encoding.

        char_encoding_type: String name for the unicode encoding that should
            be used to decode each string.

        replacement_int: The replacement codepoint to be used in place
            of invalid substrings in input.

        encoder_blocks: Number of conv blocks.

        encoder_hidden_dim: Hidden dim of conv block.

        encoder_kernel_sizes: List of kernel sizes, one for each conv block.

        encoder_depth_multiplier: Depth multiplier for depthwise conv.

        encoder_filters: Num filters for each conv block.

        encoder_dropout: Dropout rate in conv blocks.

        encoder_epsilon: Norm epsilon to use.

        encoder_activation: Activation for encoder.

        encoder_strides: List of strides, one for each conv block.

        encoder_norm_type: Norm type. One of 'layer', 'batch', or None.

        encoder_output_dim: Output encoder dimension to project encoder sequence
            outputs to. 0 to disable.

        encoder_output_activation: Activation applied onto the encoder sequence
            outputs.

        tokenizer_dense_dim: Dimension of tokenizer, applied after flattening.
            If set, expands or compresses the tokenizer to this dimension
            before the tokenizer activation is applied.

        tokenizer_activation: Activation of tokenizer layer, must
            constrain output between [-1, 1] or [0, 1].

        similarity_dim: Dimension of the similarity embedding output.
            0 to disable.

        original_decoder_size: Dimension of a single char one-hot
            auto-encoder decoder output for the original token.
            0 to disable.

        aug_decoder_size: Dimension of a single char one-hot
            auto-encoder decoder output for the augmented token.
            0 to disable.

        aug_vector_dim: Dimension of the aug vector prediction output.
            0 to disable.

        aug_matrix_dim: Dimension of the aug matrix prediction output.
            0 to disable.

        outputs_dropout_rate: Dropout rate after tokenizer layer and
            before outputs.

        outputs_norm_type: Norm used in the output heads, other than
            similarity. One of ['layer', 'batch'].

        similarity_norm_type: Norm used at the similarity output,
            one of ['layer', 'batch', 'l2', None].

    Returns:
        A CNN-based REWNet model, ready for pretraining.
    """
    inputs = layers.Input(shape=(1,), name="token", dtype=tf.string)

    # character embedding
    encoder = RetVecBinarizer(
        max_chars,
        encoding_size=char_encoding_size,
        encoding_type=char_encoding_type,
        cls_int=None,
        replacement_int=replacement_int,
        name="binarizer",
    )(inputs)

    # compress or expand char_encoding_size to encoder_hidden_dim
    if char_encoding_size != encoder_hidden_dim:
        encoder = layers.Dense(char_encoding_size)(encoder)

    for i in range(encoder_blocks):
        encoder = ConvNextBlock(
            kernel_size=encoder_kernel_sizes[i],
            depth=encoder_depth_multiplier,
            filters=encoder_filters,
            hidden_dim=encoder_hidden_dim,
            dropout_rate=encoder_dropout,
            epsilon=encoder_epsilon,
            activation=encoder_activation,
            strides=encoder_strides[i],
            residual=True,
        )(encoder)

    # intermediate layers before tokenizer
    intermediate_layer = encoder
    intermediate_layer = layers.Flatten()(intermediate_layer)

    # this is the layer is used to bound the values outputed by the tokenizer
    # between -1 and 1 using tanh, softsign etc. Allows to use activation
    # functions in the tranformers block that are unbounded such as gelu.
    # this is the layers used as output for the retvec sentence tokenizer
    # ! do not change it or the sentence tokenizer will break
    if tokenizer_dense_dim:
        tokenizer_layer = layers.Dense(
            tokenizer_dense_dim, activation=tokenizer_activation, name="tokenizer"
        )(intermediate_layer)
    else:
        tokenizer_layer = layers.Activation(
            activation=tokenizer_activation, name="tokenizer"
        )(intermediate_layer)

    # set up encoder sequence output for sequence prediction tasks
    encoder_sequence_output = encoder

    # project encoder dim if needed
    if encoder_output_dim:
        encoder_sequence_output = layers.Dense(encoder_output_dim)(
            encoder_sequence_output
        )

    if encoder_output_activation:
        encoder_sequence_output = layers.Activation(
            activation=tokenizer_activation, name="encoder_tokenizer"
        )(encoder_sequence_output)

    outputs = build_outputs(
        tokenizer_layer=tokenizer_layer,
        encoder_sequence_output=encoder_sequence_output,
        activation=encoder_activation,
        similarity_dim=similarity_dim,
        original_decoder_size=original_decoder_size,
        aug_decoder_size=aug_decoder_size,
        aug_vector_dim=aug_vector_dim,
        aug_matrix_dim=aug_matrix_dim,
        outputs_dropout_rate=outputs_dropout_rate,
        outputs_norm_type=outputs_norm_type,
        similarity_norm_type=similarity_norm_type,
    )
    return tf.keras.Model(inputs, outputs)

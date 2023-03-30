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

from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow.keras import layers

from ..layers.binarizer import RETVecBinarizer
from .layers import dense_block
from .outputs import build_outputs


def build_retvec_base_from_config(config: Dict) -> tf.keras.Model:
    m = config["model"]
    o = config["outputs"]
    return build_retvec_base(
        word_length=m["word_length"],
        char_encoding_size=m["char_encoding_size"],
        char_encoding_type=m["char_encoding_type"],
        replacement_char=m["replacement_char"],
        initial_spatial_dropout_rate=m["initial_spatial_dropout_rate"],
        projection_dims=m["projection_dims"],
        encoder_dims=m["encoder_dims"],
        encoder_dropout=m["encoder_dropout"],
        encoder_spatial_dropout_rate=m["encoder_spatial_dropout_rate"],
        encoder_initializer=m["encoder_initializer"],
        encoder_norm_type=m["encoder_norm_type"],
        encoder_norm_epsilon=m["encoder_norm_epsilon"],
        encoder_activation=m["encoder_activation"],
        encoder_seq_output_dim=m["encoder_seq_output_dim"],
        encoder_seq_output_activation=m["encoder_seq_output_activation"],
        encoder_seq_output_dropout=m["encoder_seq_output_dropout"],
        tokenizer_dense_dim=m["tokenizer_dense_dim"],
        tokenizer_activation=m["tokenizer_activation"],
        tokenizer_dropout=m["tokenizer_dropout"],
        similarity_dim=o["similarity_dim"],
        original_decoder_size=o["original_decoder_size"],
        aug_decoder_size=o["aug_decoder_size"],
        aug_vector_dim=o["aug_vector_dim"],
        aug_matrix_dim=o["aug_matrix_dim"],
        outputs_dropout_rate=o["outputs_dropout_rate"],
        similarity_norm_type=o["similarity_norm_type"],
    )


def build_retvec_base(
    word_length: int = 16,
    char_encoding_size: int = 24,
    char_encoding_type: str = "UTF-8",
    replacement_char: int = 65533,
    initial_spatial_dropout_rate: float = 0.0625,
    projection_dims: List[int] = [32, 32],
    encoder_dims: List[int] = [256],
    encoder_dropout: float = 0.0,
    encoder_spatial_dropout_rate: float = 0.0,
    encoder_initializer: str = "glorot_uniform",
    encoder_norm_type: Optional[str] = None,
    encoder_norm_epsilon: float = 1e-6,
    encoder_activation: str = "gelu",
    encoder_seq_output_dim: int = 0,
    encoder_seq_output_activation: Optional[str] = None,
    encoder_seq_output_dropout: float = 0.0,
    tokenizer_dense_dim: int = 256,
    tokenizer_activation: str = "tanh",
    tokenizer_dropout: float = 0.0,
    similarity_dim: int = 256,
    original_decoder_size: int = 0,
    aug_decoder_size: int = 0,
    aug_vector_dim: int = 0,
    aug_matrix_dim: int = 0,
    outputs_dropout_rate: float = 0.0,
    similarity_norm_type: str = "l2",
) -> tf.keras.Model:
    """RETVec model based on MLP architecture.

    Args:
        word_length: Maximum number of characters in input string to.

        char_encoding_size: Size of output character encoding.

        char_encoding_type: String name for the unicode encoding that should
            be used to decode each string.

        replacement_char: The replacement codepoint to be used in place
            of invalid substrings in input.

        initial_spatial_dropout_rate: Spatial dropout rate on character
            encoding.

        projection_dims: Dense dimensions before flatten layer.

        encoder_dims: Dense dimensions after flatten layer.

        encoder_dropout: Feature dropout rate in dense blocks.

        encoder_spatial_dropout: Spatial dropout rate in dense blocks.

        encoder_initializer: Initializer used for dense blocks in encoder.

        encoder_norm_type: Norm type. One of 'layer', 'scaled', 't5' or None.

        encoder_norm_epsilon: Epsilon value for norm.

        encoder_activation: Activation to use in encoder layers.

        encoder_seq_output_dim: Output encoder dimension to project encoder
            sequence outputs to if `encoder_sequence_pooling` is 'dense'.

        encoder_seq_output_activation: Activation applied onto the encoder
            sequence outputs.

        encoder_seq_output_dropout: Dropout on encoder seq dense layer.

        tokenizer_dense_dim: Dimension of tokenizer, applied after flattening.
            If set, expands or compresses the tokenizer to this dimension
            before the tokenizer activation is applied.

        tokenizer_activation: Activation of tokenizer layer, must
            constrain output between [-1, 1] or [0, 1].

        tokenizer_dropout: Dropout on tokenizer dense layer, if applicable.

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

        similarity_norm_type: Norm used at the similarity output,
            one of ['layer', 'batch', 'l2', None]

    Returns:
        A MLP-based RetVec model, ready for pretraining.
    """
    inputs = layers.Input(shape=(1,), name="token", dtype=tf.string)

    # character embedding
    char_encoding = RETVecBinarizer(
        word_length=word_length,
        encoding_size=char_encoding_size,
        encoding_type=char_encoding_type,
        replacement_char=replacement_char,
        name="binarizer",
    )(inputs)

    if initial_spatial_dropout_rate:
        encoder = layers.SpatialDropout1D(initial_spatial_dropout_rate)(
            char_encoding
        )

    for projection_dim in projection_dims:
        encoder = dense_block(
            x=encoder,
            units=projection_dim,
            activation=encoder_activation,
            norm_type=encoder_norm_type,
            norm_epsilon=encoder_norm_epsilon,
            dropout_rate=encoder_dropout,
        )

        if encoder_spatial_dropout_rate:
            encoder = layers.SpatialDropout1D(encoder_spatial_dropout_rate)(
                encoder
            )

    # intermediate layers before tokenizer
    encoder = layers.Flatten()(encoder)

    for encoder_dim in encoder_dims:
        encoder = dense_block(
            x=encoder,
            units=encoder_dim,
            activation=encoder_activation,
            norm_type=encoder_norm_type,
            norm_epsilon=encoder_norm_epsilon,
            dropout_rate=encoder_dropout,
            kernel_initializer=encoder_initializer,
        )

    # this is the layer is used to bound the values outputed by the tokenizer
    # between -1 and 1 using tanh, softsign etc. Allows to use activation
    # functions in the tranformers block that are unbounded such as gelu.
    # this is the layers used as output for the retvec sentence tokenizer
    # ! do not change it or the sentence tokenizer will break
    tokenizer_layer = dense_block(
        x=encoder,
        units=tokenizer_dense_dim,
        activation=tokenizer_activation,
        dropout_rate=tokenizer_dropout,
        kernel_initializer=encoder_initializer,
        name="tokenizer",
    )

    # set up encoder sequence output for sequence prediction tasks
    encoder_seq_output = encoder

    # project encoder dim if needed
    if encoder_seq_output_dim:
        encoder_seq_output = dense_block(
            x=encoder_seq_output,
            units=encoder_seq_output_dim,
            activation=encoder_seq_output_activation,
            dropout_rate=encoder_seq_output_dropout,
            kernel_initializer=encoder_initializer,
            name="encoder_seq",
        )

    outputs = build_outputs(
        tokenizer_layer=tokenizer_layer,
        encoder_seq_output=encoder_seq_output,
        activation=encoder_activation,
        similarity_dim=similarity_dim,
        original_decoder_size=original_decoder_size,
        aug_decoder_size=aug_decoder_size,
        aug_vector_dim=aug_vector_dim,
        aug_matrix_dim=aug_matrix_dim,
        outputs_dropout_rate=outputs_dropout_rate,
        similarity_norm_type=similarity_norm_type,
    )
    return tf.keras.Model(inputs, outputs)

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

from tensorflow_retvec import RetVecBinarizer

from .gau import GAU
from .layers import BertPooling, dense_block
from .outputs import build_outputs
from .positional_embeddings import (
    PositionalEmbedding,
    ScaledSinusoidalPositionalEmbedding,
)


def build_rewformer_from_config(config: Dict) -> tf.keras.Model:
    m = config["model"]
    o = config["outputs"]
    return REWformer(
        max_chars=m["max_chars"],
        char_encoding_size=m["char_encoding_size"],
        char_encoding_type=m["char_encoding_type"],
        cls_int=m["cls_int"],
        replacement_int=m["replacement_int"],
        encoder_hidden_dim=m["encoder_hidden_dim"],
        encoder_abs_pos_encoding_type=m["encoder_abs_pos_encoding_type"],
        encoder_blocks=m["encoder_blocks"],
        encoder_shared_dim=m["encoder_shared_dim"],
        encoder_expansion_factor=m["encoder_expansion_factor"],
        encoder_activation=m["encoder_activation"],
        encoder_attention_activation=m["encoder_attention_activation"],
        encoder_norm_type=m["encoder_norm_type"],
        encoder_position_encoding_type=m["encoder_position_encoding_type"],
        encoder_dropout=m["encoder_dropout"],
        encoder_attention_dropout=m["encoder_attention_dropout"],
        encoder_spatial_dropout=m["encoder_spatial_dropout"],
        encoder_epsilon=m["encoder_epsilon"],
        encoder_output_dim=m["encoder_output_dim"],
        encoder_output_activation=m["encoder_output_activation"],
        use_bert_pooling=m["use_bert_pooling"],
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


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
def REWformer(
    max_chars: int = 16,
    char_encoding_size: int = 32,
    char_encoding_type: str = "UTF-8",
    cls_int: int = 3,
    replacement_int: int = 11,
    encoder_hidden_dim: int = 128,
    encoder_abs_pos_encoding_type: str = "scaled_sin",
    encoder_blocks: int = 2,
    encoder_shared_dim: int = 32,
    encoder_expansion_factor: int = 2,
    encoder_activation: str = "swish",
    encoder_attention_activation: str = "sqrrelu",
    encoder_norm_type: str = "scaled",
    encoder_position_encoding_type: str = "rope",
    encoder_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_spatial_dropout: float = 0.0,
    encoder_epsilon: float = 1e-12,
    encoder_output_dim: int = 0,
    encoder_output_activation: str = None,
    use_bert_pooling: bool = True,
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
    """REWformer model based on transformer architecture.

    The model is based on the FLASH architecture, introduced in the paper
    Transformer Quality in Linear Time (https://arxiv.org/abs/2202.10447).

    Args:
        max_chars: Maximum number of characters to binarize. If the input
            is 2d, i.e. (batch_size, num_words) this is still the max
            characters per word.

        char_encoding_size: Size of output character encoding.

        char_encoding_type: String name for the unicode encoding that should
            be used to decode each string.

        cls_int: CLS int token to prepend to each token. Defaults to 3.

        replacement_int: The replacement codepoint to be used in place
            of invalid substrings in input.

        encoder_hidden_dim: Hidden dim of transformer block.

        encoder_abs_pos_encoding_type: Absolute positional encoding type.
            One of 'scaled_sin', 'absolute' or None.

        encoder_blocks: Number of transformer blocks.

        encoder_shared_dim: Size of shared dim in transformer blocks.

        encoder_expansion_factor: Hidden dim expansion factor.

        encoder_activation: Activation to use in projection layers.

        encoder_attention_activation: Activation to use on attention scores.

        encoder_norm_type: Norm type. One of 'layer', 'scaled', 't5' or None.

        encoder_position_encoding_type: Type of positional encoding to use.
                Either 'rope' or 'relative'.

        encoder_dropout: Feature dropout rate in transformer blocks.

        encoder_attention_dropout: Attention dropout rate in transformer
            blocks.

        encoder_spatial_dropout: Spatial dropout rate in transformer blocks.

        encoder_epsilon: Epsilon value for norm.

        encoder_output_dim: Output encoder dimension to project encoder sequence
            outputs to.

        encoder_output_activation: Activation applied onto the encoder sequence
            outputs.

        use_bert_pooling: Whether to use Bert Pooling for the tokenizer instead
            of a flatten layer.

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
            one of ['layer', 'batch', 'l2', None],

    Returns:
        A transformer-based REWNet model, ready for pretraining.
    """

    inputs = layers.Input(shape=(1,), name="token", dtype=tf.string)

    # character embedding
    encoder = RetVecBinarizer(
        max_chars,
        encoding_size=char_encoding_size,
        encoding_type=char_encoding_type,
        cls_int=cls_int,
        replacement_int=replacement_int,
        name="binarizer",
    )(inputs)

    if encoder_abs_pos_encoding_type == "scaled_sin":
        encoder = ScaledSinusoidalPositionalEmbedding(hidden_size=char_encoding_size)(encoder)

    elif encoder_abs_pos_encoding_type == "absolute":
        encoder = PositionalEmbedding()(encoder)

    # compress or expand char_encoding_size to encoder_hidden_dim
    encoder = layers.Dense(encoder_hidden_dim)(encoder)

    for _ in range(encoder_blocks):
        encoder = GAU(
            dim=encoder_hidden_dim,
            max_len=max_chars,
            shared_dim=encoder_shared_dim,
            expansion_factor=encoder_expansion_factor,
            activation=encoder_activation,
            attention_activation=encoder_attention_activation,
            position_encoding_type=encoder_position_encoding_type,
            norm_type=encoder_norm_type,
            dropout_rate=encoder_dropout,
            attention_dropout_rate=encoder_attention_dropout,
            spatial_dropout_rate=encoder_spatial_dropout,
            epsilon=encoder_epsilon,
        )(encoder)

    if use_bert_pooling:
        intermediate_layer = BertPooling()(encoder)

    else:
        intermediate_layer = layers.Flatten()(encoder)

    # this is the layer is used to bound the values outputed by the tokenizer
    # between -1 and 1 using tanh, softsign etc. Allows to use activation
    # functions in the tranformers block that are unbounded such as gelu.
    # this is the layers used as output for the retvec sentence tokenizer
    # ! do not change it or the sentence tokenizer will break
    if tokenizer_dense_dim:
        tokenizer_layer = layers.Dense(tokenizer_dense_dim, activation=tokenizer_activation, name="tokenizer")(
            intermediate_layer
        )
    else:
        tokenizer_layer = layers.Activation(activation=tokenizer_activation, name="tokenizer")(intermediate_layer)

    # set up encoder sequence output for sequence prediction tasks
    encoder_sequence_output = encoder

    # project encoder dim if needed
    if encoder_output_dim:
        encoder_sequence_output = layers.Dense(encoder_output_dim)(encoder_sequence_output)

    if encoder_output_activation:
        encoder_sequence_output = layers.Activation(activation=tokenizer_activation, name="encoder_tokenizer")(
            encoder_sequence_output
        )

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

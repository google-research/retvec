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

from typing import List

from tensorflow import Tensor
from tensorflow.keras import layers

from .layers import get_norm_layer


def build_outputs(
    tokenizer_layer: Tensor,
    encoder_seq_output: Tensor,
    activation: str,
    similarity_dim: int = 128,
    original_decoder_size: int = 0,
    aug_decoder_size: int = 0,
    aug_vector_dim: int = 0,
    aug_matrix_dim: int = 0,
    outputs_dropout_rate: float = 0.0,
    similarity_norm_type: str = "l2",
    model_type: str = "rewmlp",
) -> List[Tensor]:
    """Create output heads.

    Args:
        tokenizer_layer: Layer that will be used for tokenizing.

        encoder_sequence_output: Sequence output for encoder, if applicable.

        max_len: Maximum number of characters per word.

        activation: Activation used for all output heads, minus the last
            layers (which is fixed, i.e. softmax for complexity, etc.)

        similarity_dim: Dimension of similarity output.

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

        outputs_dropout_rate: Dropout rate to apply before output heads.

        similarity_norm_type: Norm used at the similarity output,
            one of ['layer', 'batch', 'l2', None].

    Returns:
        List of outputs for a REWNet model.
    """
    outputs = []

    if outputs_dropout_rate:
        tokenizer_layer = layers.Dropout(outputs_dropout_rate)(tokenizer_layer)
        encoder_seq_output = layers.Dropout(outputs_dropout_rate)(
            encoder_seq_output
        )

    # ! similarity must be the first output always
    similarity_output = tokenizer_layer
    similarity_dense_name = (
        "similarity" if not similarity_norm_type else "similarity_dense"
    )

    if similarity_dim:
        similarity_output = layers.Dense(
            similarity_dim, name=similarity_dense_name
        )(similarity_output)

    # output normalization for similarity head
    if similarity_norm_type:
        similarity_output = get_norm_layer(
            similarity_norm_type, name="similarity"
        )(similarity_output)

    outputs.append(similarity_output)

    # autoencoder output (decode to original token)
    if original_decoder_size:
        ori_decoder_output = layers.Dense(
            original_decoder_size, activation="sigmoid", name="ori_decoder"
        )(encoder_seq_output)
        outputs.append(ori_decoder_output)

    # autoencoder output (decode to input token)
    if aug_decoder_size:
        aug_decoder_output = layers.Dense(
            aug_decoder_size, activation="sigmoid", name="aug_decoder"
        )(encoder_seq_output)
        outputs.append(aug_decoder_output)

    # aug vector prediction output
    if aug_vector_dim:
        aug_vector_output = layers.Dense(
            aug_vector_dim, activation="sigmoid", name="aug_vector"
        )(encoder_seq_output)
        outputs.append(aug_vector_output)

    # aug matrix prediction output
    if aug_matrix_dim:
        aug_matrix_output = layers.Dense(
            aug_matrix_dim, activation="sigmoid", name="aug_matrix"
        )(encoder_seq_output)
        outputs.append(aug_matrix_output)

    return outputs

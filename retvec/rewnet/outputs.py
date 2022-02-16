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

from retvec.types import FloatTensor
from tensorflow.keras import layers

from .layers import L2Norm


def build_outputs(tokenizer_layer: FloatTensor,
                  decoder_layer: FloatTensor,
                  classification_layer: FloatTensor,
                  max_len: int,
                  tokenizer_dim: int,
                  activation: str,
                  similarity_dim: int = 256,
                  pre_complexity_dim: int = 128,
                  complexity_dim: int = 5,
                  original_decoder_size: int = 256,
                  aug_decoder_size: int = 256,
                  pre_lang_dim: int = 256,
                  lang_dim: int = 106,
                  pre_aug_vector_dim: int = 128,
                  aug_vector_dim: int = 16,
                  aug_matrix_dim: int = 16,
                  outputs_norm_type: str = 'batch',
                  similarity_norm_type: str = 'batch',
                  repeat_vector: bool = True) -> List[FloatTensor]:
    """Create output heads.

    Args:
        tokenizer_layer: Layer that will be used for tokenizing.

        decoder_layer: Layer to build decoder tasks on.

        classification_layer: Layer to build classification tasks on. This may
            be neccessary for BERT models with CLS tokens.

        max_len: Maximum number of characters per word.

        tokenizer_dim: Dimension of tokenizer, after flattening.

        activation: Activation used for all output heads, minus the last
            layers (which is fixed, i.e. softmax for complexity, etc.)

        similarity_dim: Dimension of similarity output.

        pre_complexity_dim: Dimension of the hidden dense layer in the
            complexity output. 0 to disable.

        complexity_dim: Dimension of the complexity output.
            0 to disable.

        original_decoder_size: Dimension of a single char one-hot
            auto-encoder decoder output for the original token.
            0 to disable.

        aug_decoder_size: Dimension of a single char one-hot
            auto-encoder decoder output for the augmented token.
            0 to disable.

        pre_lang_dim:  Dimension of the hidden dense layer in the
            language prediction output. 0 to disable.

        lang_dim: Dimension of the language prediction output.
            0 to disable.

        pre_aug_vector_dim: Dimension of the hidden dense layer in the
            aug vector prediction output. 0 to disable.

        aug_vector_dim: Dimension of the aug vector prediction output.
            0 to disable.

        aug_matrix_dim: Dimension of the aug matrix prediction output.
            0 to disable.

        outputs_norm_type: Norm used in the output heads, other than
            similarity. One of ['layer', 'batch'].

        similarity_norm_type: Norm used in the similarity head,
            one of ['layer', 'batch', 'l2', None]. If None or L2,
            uses batch norm for intermediate dense layers.

    Returns:
        List of outputs for a REWNet model.
    """
    outputs = []
    char_encoding_size = tokenizer_dim // max_len

    # similarity output
    # ! similarity must be the first output always!
    if similarity_dim:

        if similarity_norm_type:
            sim_pre_dense_name = 'pre_similarity'
        else:
            sim_pre_dense_name = 'similarity'

        sim_head_name = 'similarity'

        # ! l2 or no output norm defaults to use batch norm in the dense layers
        if (similarity_norm_type == 'l2' or not similarity_norm_type):
            pre_norm_type = 'batch'
        else:
            pre_norm_type = similarity_norm_type

        similarity_output = dense_output(tokenizer_layer,
                                         dim=similarity_dim,
                                         pre_dim=similarity_dim,
                                         num_layers=1,
                                         activation=activation,
                                         output_activation=None,
                                         norm_type=pre_norm_type,
                                         name=sim_pre_dense_name)
        if similarity_norm_type == 'layer':
            similarity_output = layers.LayerNormalization(
                name=sim_head_name)(similarity_output)
        elif similarity_norm_type == 'batch':
            similarity_output = layers.BatchNormalization(
                name=sim_head_name)(similarity_output)
        elif similarity_norm_type == 'l2':
            similarity_output = L2Norm(name=sim_head_name)(similarity_output)

        outputs.append(similarity_output)

    # autoencoder output (decode to original token)
    if original_decoder_size:
        decoder_output = decoder(decoder_layer,
                                 decoder_size=original_decoder_size,
                                 max_len=max_len,
                                 char_encoding_size=char_encoding_size,
                                 use_transpose=False,
                                 repeat_vector=repeat_vector,
                                 norm_type=outputs_norm_type,
                                 name='ori_decoder')
        outputs.append(decoder_output)

    # autoencoder output (decode to augmented token)
    if aug_decoder_size:
        decoder_output = decoder(decoder_layer,
                                 decoder_size=aug_decoder_size,
                                 max_len=max_len,
                                 char_encoding_size=char_encoding_size,
                                 use_transpose=False,
                                 repeat_vector=repeat_vector,
                                 norm_type=outputs_norm_type,
                                 name='aug_decoder')
        outputs.append(decoder_output)

    # complexity
    if complexity_dim:
        complexity_output = dense_output(classification_layer,
                                         dim=complexity_dim,
                                         pre_dim=pre_complexity_dim,
                                         num_layers=2,
                                         activation=activation,
                                         output_activation='softmax',
                                         norm_type=outputs_norm_type,
                                         name='complexity')
        outputs.append(complexity_output)

    # language output
    if lang_dim:
        lang_output = dense_output(classification_layer,
                                   dim=lang_dim,
                                   pre_dim=pre_lang_dim,
                                   num_layers=2,
                                   activation=activation,
                                   output_activation='softmax',
                                   norm_type=outputs_norm_type,
                                   name='lang')
        outputs.append(lang_output)

    if aug_vector_dim:
        # augmentation vector output
        aug_vector_output = dense_output(classification_layer,
                                         dim=aug_vector_dim,
                                         pre_dim=pre_aug_vector_dim,
                                         num_layers=2,
                                         activation=activation,
                                         output_activation='softmax',
                                         norm_type=outputs_norm_type,
                                         name='aug_vector')
        outputs.append(aug_vector_output)

    if aug_matrix_dim:
        # augmentation matrix output
        aug_matrix_output = conv_output(decoder_layer,
                                        max_len=max_len,
                                        char_encoding_size=char_encoding_size,
                                        dim=aug_matrix_dim,
                                        num_layers=2,
                                        activation=activation,
                                        output_activation='softmax',
                                        filter_size=16,
                                        kernel_size=3,
                                        compression_rate=2,
                                        norm_type=outputs_norm_type,
                                        name='aug_matrix')
        outputs.append(aug_matrix_output)
    return outputs


def dense_output(x: FloatTensor,
                 dim: int,
                 pre_dim: int,
                 num_layers: int,
                 activation: str,
                 output_activation: str,
                 norm_type: str,
                 name: str,
                 compression_factor: int = 1,
                 epsilon: float = 1e-6) -> FloatTensor:
    """Convolutional output head. """

    output = x

    if pre_dim:
        for i in range(num_layers):
            layer_name = f"pre_{name}{i}"
            output = layers.Dense(pre_dim,
                                  name=layer_name)(output)

            if norm_type == 'layer':
                output = layers.LayerNormalization(
                    epsilon=epsilon, name=layer_name + '_layernorm')(output)

            elif norm_type == 'batch':
                output = layers.BatchNormalization(
                    name=layer_name + '_batchnorm')(output)

            output = layers.Activation(activation,
                                       name=layer_name + '_act')(output)

            pre_dim //= compression_factor

    output = layers.Dense(dim,
                          activation=output_activation,
                          name=name)(output)
    return output


def decoder(x: FloatTensor,
            decoder_size: int,
            max_len: int,
            char_encoding_size: int,
            num_layers: int = 2,
            activation: str = 'swish',
            initial_filters: int = 64,
            expansion_rate: int = 2,
            kernel_size: int = 5,
            use_transpose: bool = False,
            norm_type: str = 'layer',
            epsilon: float = 1e-6,
            repeat_vector: bool = True,
            name: str = 'ori_decoder') -> FloatTensor:
    """Autoencoder decoder head

    Args:
        x: Last encoding layer.
        decoder_size: Size of single char one-hot decoder.
        num_layers: Number of conv layers (excluding
            last output layer).
        activation: Activation function.
        max_len: Num chars to decode.
        batch_norm: Use batch normalization.
        char_encoding_size: dimension of single char encoding.
        name: name of the decoder, since there may be multiple.
        repeat_vector: Use repeat vector instead of reshape.
    Returns:
        Decoder outputs [max_len, decoder_size]

    """
    decoder = x

    if repeat_vector:
        decoder = layers.RepeatVector(max_len)(decoder)
    else:
        decoder = layers.Reshape((max_len, char_encoding_size))(decoder)

    filters = initial_filters
    for i in range(num_layers):
        layer_name = f"pre_{name}{i}"

        if use_transpose:
            decoder = layers.Conv1DTranspose(filters,
                                             kernel_size,
                                             padding='same',
                                             name=layer_name)(decoder)
        else:
            decoder = layers.Conv1D(filters,
                                    kernel_size,
                                    padding='same',
                                    name=layer_name)(decoder)

        if norm_type == 'layer':
            decoder = layers.LayerNormalization(
                epsilon=epsilon, name=layer_name + '_layernorm')(decoder)

        elif norm_type == 'batch':
            decoder = layers.BatchNormalization(
                name=layer_name + '_batchnorm')(decoder)

        decoder = layers.Activation(activation,
                                    name=layer_name + '_act')(decoder)

        filters = filters * expansion_rate

    decoder_output: FloatTensor = layers.Conv1D(decoder_size,
                                                3,
                                                activation='softmax',
                                                padding='same',
                                                name=name)(decoder)

    return decoder_output


def conv_output(x: FloatTensor,
                max_len: int,
                char_encoding_size: int,
                dim: int,
                num_layers: int,
                activation: str,
                output_activation: str,
                filter_size: int = 16,
                kernel_size: int = 3,
                compression_rate: int = 2,
                norm_type: str = 'layer',
                epsilon: float = 1e-6,
                name: str = 'aug_matrix') -> FloatTensor:
    """Convolutional output head. """

    output = layers.Reshape((max_len, char_encoding_size))(x)
    filters = filter_size

    for i in range(num_layers):
        layer_name = f"pre_{name}{i}"
        output = layers.Conv1D(filters,
                               kernel_size,
                               padding='same',
                               name=layer_name)(output)

        if norm_type == 'layer':
            output = layers.LayerNormalization(
                epsilon=epsilon, name=layer_name + '_layernorm')(output)

        elif norm_type == 'batch':
            output = layers.BatchNormalization(
                name=layer_name + '_batchnorm')(output)

        output = layers.Activation(activation,
                                   name=layer_name + '_act')(output)

        filters //= compression_rate

    conv_output: FloatTensor = layers.Conv1D(dim,
                                             kernel_size,
                                             activation=output_activation,
                                             padding='same',
                                             name=name)(output)
    return conv_output

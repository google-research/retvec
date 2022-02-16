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
from retvec import RecVec
from retvec.projection import FourierFeatureProjection
from tensorflow.keras import layers
from tensorflow_similarity.models import SimilarityModel

from .layers import TBlock
from .outputs import build_outputs
from .t5 import T5Block, T5LayerNorm


def build_rewbert_from_config(c: Dict) -> tf.keras.Model:
    m = c['models']['rewbert']
    o = c['outputs']
    return REWBert(max_len=c['max_len'],
                   char_embedding_size=c['char_embedding_size'],
                   projection_dim=m.get('projection_dim', 0),
                   char_encoding_size=m['char_encoding_size'],
                   encoder_blocks=m['encoder_blocks'],
                   encoder_heads=m['encoder_heads'],
                   encoder_hidden_dim=m['encoder_hidden_dim'],
                   encoder_head_dim=m['encoder_head_dim'],
                   encoder_dropout=m['encoder_dropout'],
                   encoder_spatial_dropout=m.get('encoder_spatial_dropout', 0),
                   encoder_gated_ffn=m['encoder_gated_ffn'],
                   epsilon=m['epsilon'],
                   encoder_out_dim=m['encoder_out_dim'],
                   encoder_activation=m['encoder_activation'],
                   encoder_use_t5=m['encoder_use_t5'],
                   encoder_buckets=m['encoder_buckets'],
                   encoder_max_distance=m['encoder_max_distance'],
                   tokenizer_dim=m['tokenizer_dim'],
                   tokenizer_dropout_rate=m.get('tokenizer_dropout_rate', 0),
                   tokenizer_activation=m['tokenizer_activation'],
                   similarity_dim=o['similarity_dim'],
                   original_decoder_size=o['original_decoder_size'],
                   aug_decoder_size=o['aug_decoder_size'],
                   complexity_dim=o.get('complexity_dim', 0),
                   lang_dim=o.get('lang_dim', 0),
                   aug_vector_dim=o.get('aug_vector_dim', 0),
                   aug_matrix_dim=o.get('aug_matrix_dim', 0),
                   outputs_norm_type=o.get('outputs_norm_type', 'batch'),
                   similarity_norm_type=o.get('similarity_norm_type', 'batch'),
                   repeat_vector=o.get('repeat_vector', True))


@tf.keras.utils.register_keras_serializable(package="retvec")
def REWBert(max_len: int = 16,

            # char embedding params
            char_embedding_size: int = 16,

            # projection
            projection_dim: int = 0,
            projection_scale: int = 10,
            projection_activations: List[str] = ['sin', 'softsign'],
            projection_use_circle: bool = False,

            # encoder params
            char_encoding_size: int = 64,
            encoder_blocks: int = 2,
            encoder_heads: int = 8,
            encoder_hidden_dim: int = 128,
            encoder_head_dim: int = 32,
            encoder_dropout: float = 0.05,
            encoder_spatial_dropout: float = 0.05,
            encoder_gated_ffn: bool = True,
            epsilon: float = 1e-12,
            encoder_out_dim: int = None,
            encoder_activation: str = "swish",
            encoder_use_t5: bool = True,
            encoder_buckets: int = 16,
            encoder_max_distance: int = 16,

            # tokenizer
            tokenizer_dim: int = 256,
            tokenizer_dropout_rate: float = 0.0,
            tokenizer_activation: str = "softsign",

            # outputs
            similarity_dim: int = 256,
            original_decoder_size: int = 256,
            aug_decoder_size: int = 256,
            complexity_dim: int = 0,
            lang_dim: int = 0,
            aug_vector_dim: int = 0,
            aug_matrix_dim: int = 0,
            outputs_norm_type: str = 'batch',
            similarity_norm_type: str = 'batch',
            repeat_vector: bool = True) -> tf.keras.Model:
    """REWNet model based on T5 transformer architecture.

    Args:
        max_len: Maximum number of characters per word.

        char_embedding_size: Size of RecVec character embedding.

        projection_dim: Fourier feature projection layer dimension.
            0 to disable projection and use normal stem instead.

        projection_scale: Scale of gaussian kernel in projection
            layer.

        projection_activations: List of activation functions to
            apply on the projection.

        projection_use_circle: Whether to use circle projection.

        projection_norm: Norm to use in projection stem, one of
            ['layer', 'batch', 't5'].

        char_encoding_size: Compress or expand from `char_embedding_size`
            to `char_encoding_size` in the stem. Equal to the dim
            of the transformer encoder blocks.

        encoder_blocks: Number of transformer encoder blocks.

        encoder_heads: Number of heads per block.

        encoder_hidden_dim: Hidden dim of transformer blocks.

        encoder_head_dim: Dimension of heads in encoder blocks.

        encoder_dropout: Dropout to use in encoder blocks.

        encoder_gated_ffn: Whether to use gated ffn.

        epsilon: Epsilon in layernorm in encoder.

        encoder_out_dim: Dim to compress to in last encoder block,
            no compression if None.

        encoder_activation: Activation for the encoder.

        encoder_use_t5: Use T5 transformer blocks instead of vanilla blocks.

        encoder_buckets: Number of position buckets in relative attention.

        encoder_max_distance: Max distance in relative attention.

        tokenizer_dim: Dimension of tokenizer, applied after flattening.

        tokenizer_dropout_rate: Dropout before tokenizer layer.

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

        complexity_dim: Dimension of the complexity output.
            0 to disable.

        lang_dim: Dimension of the language prediction output.
            0 to disable.

        aug_vector_dim: Dimension of the aug vector prediction output.
            0 to disable.

        aug_matrix_dim: Dimension of the aug matrix prediction output.
            0 to disable.

        outputs_norm_type: Norm used in the output heads, other than
            similarity. One of ['layer', 'batch'].

        similarity_norm_type: Norm used in the similarity head,
            one of ['layer', 'batch', 'l2', None]. If None or L2,
            uses batch norm for intermediate dense layers.

        repeat_vector: Whether to use repeat vector in decoding.

    Notes:
        The tokenizer activation must constrain the output between -1 and 1
        or 0 and 1 to be usable as a tokenizer.

        Architecture inspired by T5 transformer
        https://arxiv.org/abs/1910.10683
    """
    # Additional variables, that are unlikly to be tweaked
    cls_int = None  # ! CLS token doesn't seem to help
    char_lower_case = False  # we don't want to lower_case
    char_masks = 0
    char_folds = 0

    if not encoder_out_dim:
        encoder_out_dim = char_encoding_size

    inputs = layers.Input(shape=(1, ), name="token", dtype=tf.string)

    # character embedding
    encoder = RecVec(
        max_len,
        embedding_size=char_embedding_size,
        masks=char_masks,
        folds=char_folds,
        lower_case=char_lower_case,
        cls_int=cls_int)(inputs)

    if projection_dim:
        encoder = FourierFeatureProjection(
            gaussian_projection_dim=projection_dim,
            gaussian_scale=projection_scale,
            activations=projection_activations,
            circle_projection=projection_use_circle)(encoder)

        if encoder_use_t5:
            encoder = T5LayerNorm(epsilon=epsilon)(encoder)
        else:
            encoder = layers.LayerNormalization(epsilon=epsilon)(encoder)

    else:
        # compress or expand char_embedding_size to char_encoding_size
        encoder = layers.Dense(char_encoding_size,
                               activation=encoder_activation,
                               name='encoder_start')(encoder)

        if encoder_use_t5:
            encoder = T5LayerNorm(epsilon=epsilon)(encoder)
        else:
            encoder = layers.LayerNormalization(epsilon=epsilon)(encoder)

        encoder = layers.Activation(encoder_activation)(encoder)

    for _ in range(encoder_blocks - 1):
        if encoder_use_t5:
            encoder = T5Block(dim=char_encoding_size,
                              hidden_dim=encoder_hidden_dim,
                              out_dim=char_encoding_size,
                              num_heads=encoder_heads,
                              head_dim=encoder_head_dim,
                              dropout_rate=encoder_dropout,
                              spatial_dropout_rate=encoder_spatial_dropout,
                              activation=encoder_activation,
                              epsilon=epsilon,
                              use_gated_ffn=encoder_gated_ffn,
                              position_buckets=encoder_buckets,
                              position_max_distance=encoder_max_distance,
                              )(encoder)
        else:
            encoder = TBlock(dim=char_encoding_size,
                             hidden_dim=encoder_hidden_dim,
                             out_dim=char_encoding_size,
                             num_heads=encoder_heads,
                             head_dim=encoder_head_dim,
                             dropout_rate=encoder_dropout,
                             spatial_dropout_rate=encoder_spatial_dropout,
                             activation=encoder_activation,
                             epsilon=epsilon,
                             use_gated_ffn=encoder_gated_ffn,
                             )(encoder)

    # last block separated since we may compress the out dimension
    if encoder_use_t5:
        encoder = T5Block(dim=char_encoding_size,
                          hidden_dim=encoder_hidden_dim,
                          out_dim=encoder_out_dim,
                          num_heads=encoder_heads,
                          head_dim=encoder_head_dim,
                          dropout_rate=encoder_dropout,
                          spatial_dropout_rate=encoder_spatial_dropout,
                          activation=encoder_activation,
                          epsilon=epsilon,
                          use_gated_ffn=encoder_gated_ffn,
                          position_buckets=encoder_buckets,
                          position_max_distance=encoder_max_distance,
                          )(encoder)
    else:
        encoder = TBlock(dim=char_encoding_size,
                         hidden_dim=encoder_hidden_dim,
                         out_dim=encoder_out_dim,
                         num_heads=encoder_heads,
                         head_dim=encoder_head_dim,
                         dropout_rate=encoder_dropout,
                         spatial_dropout_rate=encoder_spatial_dropout,
                         activation=encoder_activation,
                         epsilon=epsilon,
                         use_gated_ffn=encoder_gated_ffn,
                         )(encoder)

    # intermediate layers before tokenizer
    intermediate_layer = encoder
    intermediate_layer = layers.Flatten()(intermediate_layer)

    if tokenizer_dropout_rate:
        intermediate_layer = layers.Dropout(
            tokenizer_dropout_rate)(intermediate_layer)

    # this is the layer is used to bound the values outputed by the tokenizer
    # between -1 and 1 using tanh, softsign etc. Allows to use activation
    # functions in the tranformers block that are unbounded such as gelu.
    # this is the layers used as output for the retvec sentence tokenizer
    # ! do not change it or the sentence tokenizer will break
    tokenizer_layer = layers.Dense(tokenizer_dim,
                                   activation=tokenizer_activation,
                                   name='tokenizer')(intermediate_layer)

    if outputs_norm_type == 'batch':
        tokenizer_layer = layers.BatchNormalization()(tokenizer_layer)

    elif outputs_norm_type == 'layer':
        tokenizer_layer = layers.LayerNormalization()(tokenizer_layer)

    outputs = build_outputs(
        # input layers
        tokenizer_layer=tokenizer_layer,
        decoder_layer=tokenizer_layer,
        classification_layer=tokenizer_layer,

        # base params
        max_len=max_len,
        tokenizer_dim=tokenizer_dim,
        activation=encoder_activation,

        # output heads
        similarity_dim=similarity_dim,
        original_decoder_size=original_decoder_size,
        aug_decoder_size=aug_decoder_size,
        complexity_dim=complexity_dim,
        lang_dim=lang_dim,
        aug_vector_dim=aug_vector_dim,
        aug_matrix_dim=aug_matrix_dim,
        outputs_norm_type=outputs_norm_type,
        similarity_norm_type=similarity_norm_type,
        repeat_vector=repeat_vector
    )

    if similarity_dim:
        return SimilarityModel(inputs, outputs)
    else:
        return tf.keras.Model(inputs, outputs)

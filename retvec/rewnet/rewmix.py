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

import tensorflow as tf
from retvec import RecVec
from retvec.types import FloatTensor
from tensorflow.keras import layers
from tensorflow_similarity.models import SimilarityModel

from .outputs import build_outputs


def build_rewmix_from_config(c: Dict) -> tf.keras.Model:
    m = c['models']['rewmix']
    o = c['outputs']
    return REWMix(max_len=c['max_len'],
                  dropout_rate=m['dropout_rate'],
                  batch_norm=m['batch_norm'],
                  char_embedding_size=c['char_embedding_size'],
                  num_blocks=m['num_blocks'],
                  encoder_dim=m['encoder_dim'],
                  encoder_hidden_dim=m['encoder_hidden_dim'],
                  encoder_combine_filters=m['encoder_combine_filters'],
                  encoder_activation=m['encoder_activation'],
                  tokenizer_dim=m['tokenizer_dim'],
                  tokenizer_activation=m['tokenizer_activation'],
                  similarity_dim=o['similarity_dim'],
                  original_decoder_size=o['original_decoder_size'],
                  aug_decoder_size=o['aug_decoder_size'],
                  complexity_dim=o['complexity_dim'],
                  lang_dim=o['lang_dim'],
                  aug_vector_dim=o['aug_vector_dim'],
                  aug_matrix_dim=o['aug_matrix_dim'],
                  outputs_norm_type=o['outputs_norm_type'],
                  similarity_norm_type=o['similarity_norm_type'])


@tf.keras.utils.register_keras_serializable(package="retvec")
class MlpBlock(layers.Layer):
    def __init__(self, in_dim: int, out_dim: int, activation: str) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        self.Dense1 = layers.Dense(in_dim)
        self.Activation = layers.Activation(activation)
        self.Dense2 = layers.Dense(out_dim)

    def call(self, x: FloatTensor) -> FloatTensor:
        x = self.Dense1(x)
        x = self.Activation(x)
        x = self.Dense2(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "activation": self.activation,
            "out_dim": self.out_dim
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
class MixerBlock(layers.Layer):
    def __init__(self,
                 max_len: int,
                 encoder_dim: int,
                 hidden_dim: int,
                 activation: str) -> None:
        super().__init__()
        self.max_len = max_len
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # sequence encoder
        self.LayerNormC = layers.LayerNormalization()
        self.PermuteC = layers.Permute((2, 1))
        self.MlpC = MlpBlock(max_len, max_len, activation)
        self.PermuteS = layers.Permute((2, 1))

        # sequence encoder
        self.LayerNormS = layers.LayerNormalization()
        self.MlpS = MlpBlock(hidden_dim, encoder_dim, activation)

    def call(self, x: FloatTensor) -> FloatTensor:

        # channel encoder
        residual = x
        x = self.LayerNormC(x)
        x = self.PermuteC(x)
        x = self.MlpC(x)
        x = self.PermuteS(x)
        x = x + residual

        # sequence encoder
        residual = x
        x = self.LayerNormS(x)
        x = self.MlpS(x)
        x = x + residual
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_len": self.max_len,
            "encoder_dim": self.encoder_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
def REWMix(max_len: int = 16,

           # char embedding params
           char_embedding_size: int = 16,

           # encoder params
           num_blocks: int = 8,
           encoder_dim: int = 16,
           encoder_hidden_dim: int = 128,
           encoder_combine_filters: int = 16,
           encoder_activation: str = "swish",

           # tokenizer
           tokenizer_dim: int = 256,
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
           similarity_norm_type: str = 'batch') -> tf.keras.Model:
    """REWNET model based on MLP mixer architecture.

    Args:
        max_len: Maximum number of characters per word.

        dropout_rate: Dropout rate to be applied after the encoder.

        batch_norm: Whether to use batch norm after the encoder.

        char_embedding_size: Size of RecVec character embedding.

        num_blocks: Number of mixer blocks.

        encoder_dim: Mixer block dimension.

        encoder_hidden_dim: Mixer block hidden dimension.

        encoder_combine_filters: Number of filters to combine with
            between the encoder and tokenizer layer. If equal to
            `encoder_filters`, disables the combine layer.

        encoder_activation: Activation for the encoder.

        tokenizer_dim: Dimension of tokenizer, applied after flattening.

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

    Notes:
        The tokenizer activation must constrain the output between -1 and 1
        or 0 and 1 to be usable as a tokenizer.

        Architecture inspired by MLP-Mixer: https://arxiv.org/abs/2105.01601
    """
    # Additional variables, that are unlikly to be tweaked
    cls_int = None  # no need of a CLS char for Mixer models
    char_lower_case = False  # we don't want to lower_case
    char_masks = 0
    char_folds = 0
    encoder_combine_kernel_size = 3

    inputs = layers.Input(shape=(1, ), name="token", dtype=tf.string)
    encoder = RecVec(
        max_len,
        embedding_size=char_embedding_size,
        masks=char_masks,
        folds=char_folds,
        lower_case=char_lower_case,
        cls_int=cls_int)(inputs)

    # stem
    encoder = layers.Dense(encoder_dim, name='stem')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(encoder_activation)(encoder)

    # mixing blocks
    for _ in range(num_blocks):
        encoder = MixerBlock(max_len=max_len,
                             encoder_dim=encoder_dim,
                             hidden_dim=encoder_hidden_dim,
                             activation=encoder_activation)(encoder)

    # intermediate layers before tokenizer
    intermediate_layer = encoder
    if encoder_combine_filters != encoder_dim:
        intermediate_layer = layers.Conv1D(encoder_combine_filters,
                                           encoder_combine_kernel_size,
                                           activation=encoder_activation,
                                           padding='same')(intermediate_layer)
    intermediate_layer = layers.Flatten()(intermediate_layer)

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
    )

    if similarity_dim:
        return SimilarityModel(inputs, outputs)
    else:
        return tf.keras.Model(inputs, outputs)

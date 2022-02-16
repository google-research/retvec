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

import tensorflow as tf
from retvec import RecVec
from retvec.projection import FourierFeatureProjection
from retvec.types import FloatTensor
from tensorflow.keras import layers
from tensorflow_similarity.models import SimilarityModel

from .outputs import build_outputs


def build_rewcnn_from_config(c: Dict) -> tf.keras.Model:
    m = c['models']['rewcnn']
    o = c['outputs']
    return REWCNN(max_len=c['max_len'],
                  char_embedding_size=c['char_embedding_size'],
                  projection_dim=m.get('projection_dim', 0),
                  spatial_dropout_rate=m.get('spatial_dropout_rate', 0),
                  feature_dropout_rate=m.get('feature_dropout_rate', 0),
                  encoder_initial_kernel_size=m['encoder_initial_kernel_size'],
                  encoder_filters=m['encoder_filters'],
                  encoder_kernel_size_1=m['encoder_kernel_size_1'],
                  expansion_1=m['expansion_1'],
                  encoder_kernel_size_2=m['encoder_kernel_size_2'],
                  expansion_2=m['expansion_2'],
                  use_fused_mbconv=m['use_fused_mbconv'],
                  encoder_activation=m['encoder_activation'],
                  encoder_out_dim=m['encoder_out_dim'],
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
class SEBlock(layers.Layer):
    def __init__(self, filters: int, se_ratio: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.se_ratio = se_ratio

        self.pool = layers.Conv1D(1, 1, use_bias=False, padding='same')
        self.squeeze = layers.Conv1D(se_ratio * filters,
                                     kernel_size=1,
                                     padding='same',
                                     activation='relu')
        self.excite = layers.Conv1D(
            filters, kernel_size=1, padding='same', activation='hard_sigmoid')

    def call(self, inputs: FloatTensor, training: bool) -> FloatTensor:
        x = self.pool(inputs)
        x = self.squeeze(x)
        x = self.excite(x)
        return x * inputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "se_ratio": self.se_ratio
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
class FusedMBConvBlock(layers.Layer):
    """Fused Inverted residual block from Effnet v2"""

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 activation: str,
                 expansion: int,
                 kernel_size: int,
                 strides: int = 1,
                 se_ratio: float = 0,
                 residual: bool = True,
                 dropout_rate: float = 0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.expansion = expansion
        self.activation = activation
        self.dropout_rate = dropout_rate

        if not residual or not out_filters == in_filters or strides > 1:
            self.residual = False
        else:
            self.residual = True

        # conv shortcut
        self.convshort = layers.Conv1D(self.out_filters, kernel_size=1,
                                       padding='valid', use_bias=False)

        # expand
        pad = "same" if strides == 1 else 'valid'
        self.convexpand = layers.Conv1D(self.in_filters * self.expansion,
                                        kernel_size=kernel_size,
                                        strides=self.strides,
                                        use_bias=False,
                                        padding=pad)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation)

        self.dropout = layers.Dropout(self.dropout_rate)

        self.se_block = SEBlock(self.in_filters * self.expansion,
                                self.se_ratio)
        # project
        self.convproject = layers.Conv1D(self.out_filters, kernel_size=1,
                                         padding='valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(activation)

        self.add = layers.Add()

    def call(self, inputs: FloatTensor, training: bool) -> FloatTensor:
        shortcut = inputs

        x = self.convexpand(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        if self.dropout_rate and self.expansion > 1:
            x = self.dropout(x)

        if self.se_ratio:
            x = self.se_block(x)

        x = self.convproject(x)
        x = self.bn2(x, training=training)

        if self.residual:
            x = self.add([x, shortcut])
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_filters": self.in_filters,
            "out_filters": self.out_filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "residual": self.residual,
            "expansion": self.expansion,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
class MBConvBlock(layers.Layer):
    """MBConv: Mobile Inverted Residual Bottleneck from Effnet v2"""

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 activation: str,
                 expansion: int,
                 kernel_size: int,
                 strides: int = 1,
                 se_ratio: float = 0,
                 residual: bool = True,
                 dropout_rate: float = 0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.expansion = expansion
        self.activation = activation
        self.dropout_rate = dropout_rate

        if not residual or not out_filters == in_filters or strides > 1:
            self.residual = False
        else:
            self.residual = True

        # expand
        pad = "same" if strides == 1 else 'valid'
        self.convexpand = layers.Conv1D(self.in_filters * self.expansion,
                                        kernel_size=kernel_size,
                                        strides=self.strides,
                                        use_bias=False,
                                        padding=pad)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(activation)

        self.depthconv = layers.DepthwiseConv1D(kernel_size=kernel_size,
                                                strides=self.strides,
                                                padding=pad,
                                                use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(activation)

        self.dropout = layers.Dropout(self.dropout_rate)

        self.se_block = SEBlock(self.in_filters * self.expansion,
                                self.se_ratio)

        # project
        self.convproject = layers.Conv1D(self.out_filters, kernel_size=1,
                                         padding='valid', use_bias=False)

        self.bn3 = layers.BatchNormalization()
        self.add = layers.Add()

    def call(self, inputs: FloatTensor, training: bool) -> FloatTensor:
        shortcut = inputs

        x = self.convexpand(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.depthconv(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        if self.dropout_rate and self.expansion > 1:
            x = self.dropout(x)

        if self.se_ratio:
            x = self.se_block(x)

        x = self.convproject(x)
        x = self.bn3(x, training=training)

        if self.residual:
            x = self.add([x, shortcut])
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_filters": self.in_filters,
            "out_filters": self.out_filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "residual": self.residual,
            "expansion": self.expansion,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }


@tf.keras.utils.register_keras_serializable(package="retvec")
def REWCNN(max_len: int = 16,

           # char embedding
           char_embedding_size: int = 16,

           # projection
           projection_dim: int = 0,
           projection_scale: int = 10,
           projection_activations: List[str] = ['sin', 'softsign'],
           projection_use_circle: bool = False,

           # stem
           spatial_dropout_rate: float = 0.0,
           feature_dropout_rate: float = 0.0,

           # encoder
           encoder_initial_kernel_size: int = 9,
           encoder_filters: int = 64,
           encoder_kernel_size_1: int = 3,
           expansion_1: int = 4,
           encoder_kernel_size_2: int = 3,
           expansion_2: int = 4,
           use_fused_mbconv: bool = True,
           encoder_activation: str = "swish",
           encoder_out_dim: int = None,

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
    """REWNet model based on CNN architecture.

    Args:
        max_len: Maximum number of characters per word.

        char_embedding_size: Size of RecVec character embedding.

        spatial_dropout_rate: Spatial dropout rate at stem.

        feature_dropout_rate: Feature dropout rate at stem.

        encoder_initial_kernel_size: Kernel size of stem conv layer.

        encoder_filters: Number of filters for the conv blocks.

        encoder_kernel_size_1: Kernel size of first conv block.

        expansion_1: Expansion rate of first conv block.

        encoder_kernel_size_2: Kernel size of second conv block.

        expansion_2: Expansion rate of second conv block.

        use_fused_mbconv: Use fused MB conv block instead of
            regular MB conv block.

        encoder_activation: Activation for the encoder.

        encoder_out_dim: Dim to compress to in last encoder block,
            no compression if None.

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

    Notes:
        The tokenizer activation must constrain the output between -1 and 1
        or 0 and 1 to be usable as a tokenizer.

        Architecture inspired by MobileNetV2: https://arxiv.org/abs/1801.04381
    """
    # Additional variables, that are unlikly to be tweaked
    cls_int = None  # no CLS char for CNN models
    char_lower_case = False  # we don't want to lower_case
    char_masks = 0
    char_folds = 0

    inputs = layers.Input(shape=(1, ), name="token", dtype=tf.string)

    # character embedding
    encoder = RecVec(max_len,
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
        encoder = layers.BatchNormalization()(encoder)

    else:
        # stem
        encoder = layers.Conv1D(filters=encoder_filters,
                                kernel_size=encoder_initial_kernel_size,
                                strides=1,
                                use_bias=False, padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation(encoder_activation)(encoder)

    if spatial_dropout_rate:
        encoder = layers.SpatialDropout1D(spatial_dropout_rate)(encoder)

    if feature_dropout_rate:
        encoder = layers.Dropout(feature_dropout_rate)(encoder)

    block = FusedMBConvBlock if use_fused_mbconv else MBConvBlock

    # first IR block
    encoder = block(in_filters=encoder_filters,
                    out_filters=encoder_filters,
                    kernel_size=encoder_kernel_size_1,
                    activation=encoder_activation,
                    expansion=expansion_1)(encoder)

    # second IR block
    encoder = block(in_filters=encoder_filters,
                    out_filters=encoder_filters,
                    kernel_size=encoder_kernel_size_2,
                    activation=encoder_activation,
                    expansion=expansion_2)(encoder)

    # intermediate layers before tokenizer
    intermediate_layer = encoder

    # compression to encoder_out_dim
    if encoder_out_dim:
        intermediate_layer = layers.Conv1D(encoder_out_dim,
                                           kernel_size=3,
                                           padding='same',
                                           use_bias=False)(intermediate_layer)
        intermediate_layer = layers.BatchNormalization()(intermediate_layer)
        intermediate_layer = layers.Activation(
            encoder_activation)(intermediate_layer)

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

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
from tensorflow.keras import layers

from retvec import RecVec
from retvec.types import FloatTensor, Tensor

from .projection import FourierFeatureProjection


@tf.keras.utils.register_keras_serializable(package="retvec")
class RetVec(tf.keras.layers.Layer):
    "RetVec: Resilient and Efficient Text Vectorizer"

    def __init__(self,
                 model: str,
                 max_len: int = 128,
                 sep: str = '',
                 eager: bool = False,
                 verbose: bool = False,
                 trainable: bool = False,

                 # projection
                 projection_dim: int = 0,
                 gaussian_scale: float = 10.0,
                 activations: List = ['softsign'],
                 circle_projection: bool = False,
                 merge: str = "concat",
                 dropout_rate: float = 0.0,
                 spatial_dropout_rate: float = 0.0,
                 norm_type: str = 'batch',

                 **kwargs) -> None:
        """Build a RetVec model.

        Args:
            model: Path to saved REW* model. If None, will default to using no
                model (just the char embedding).

            max_len: Max length of text split by `sep`.

            sep: Separator to split on.

            eager: Use eager mode.

            verbose: Verbosity.

            trainable: Whether to set this model to be trainable.

            projection_dim: Gaussian kernel projected dimension.
                If zero, uses default RecVec character encoding. The
                projection is only used if `model` is None.

            gaussian_scale: Scale of the gaussian kernel in fourier feature
                projection layer.

            activations: List of functions to apply on the gaussian projection.

            circle_projection: Whether to use circle projection.

            merge: Merge strategy of projections, one of ['concat', 'add',
                or 'mul'].

            dropout_rate: Dropout rate to use after projection.

            spatial_dropout_rate: Spatial dropout rate to use after projection.

            norm_type: norm type to apply after projection. One of 'batch',
                'layer' or None.

        """
        super(RetVec, self).__init__()
        self.model = model
        self.max_len = max_len
        self.sep = sep
        self.eager = eager
        self.verbose = verbose
        self.trainable = trainable
        self.projection_dim = projection_dim
        self.gaussian_scale = gaussian_scale
        self.activations = activations
        self.circle_projection = circle_projection
        self.merge = merge
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.norm_type = norm_type

        if not model:

            self.ce = RecVec(**kwargs)
            # used for reshaping in call
            outsize = self.ce.embedding_size * self.ce.max_len
            self.ce_size = tf.constant(outsize, dtype='int32')

            if self.projection_dim:
                self.proj = FourierFeatureProjection(
                    gaussian_projection_dim=self.projection_dim,
                    gaussian_scale=self.gaussian_scale,
                    activations=self.activations,
                    circle_projection=self.circle_projection,
                    merge=self.merge
                )
            if self.norm_type == 'batch':
                self.norm = layers.BatchNormalization()
            elif self.norm_type == 'layer':
                self.norm = layers.LayerNormalization()

            self.dropout = layers.Dropout(self.dropout_rate)
            self.spatial_drop = layers.SpatialDropout1D(
                self.spatial_dropout_rate)

        else:
            # print summary
            if verbose:
                print("|-Model", self.model)

            # initializing model and layers
            self.rewnet = self._load(model)

    @tf.function()  # jit_compile=True)
    def call(self, inputs: Tensor, training: bool = False) -> FloatTensor:
        inputs = tf.stop_gradient(inputs)

        rtensor = tf.strings.split(inputs, sep=self.sep,
                                   maxsplit=self.max_len)

        # # the shape here is different than in eager because you
        # havce batch, 1 sentence, nword so we need to pass and then remove it
        if self.eager:
            stensor = rtensor.to_tensor(default_value='',
                                        shape=(rtensor.shape[0],
                                               self.max_len))
        else:
            tensor = rtensor.to_tensor(default_value='',
                                       shape=(rtensor.shape[0], 1,
                                              self.max_len))
            stensor = tf.squeeze(tensor, axis=1)

        if self.model:
            output = self._model_inference(stensor, tf.shape(stensor)[0],
                                           self.max_len, training)
        else:
            # no model
            output = self._no_model_inference(stensor, tf.shape(stensor)[0],
                                              self.max_len, self.ce_size,
                                              training)

        output_tensor: FloatTensor = output
        return output_tensor

    # ! experimental_compile doesn't work
    def _model_inference(self,
                         tensor: Tensor,
                         batch_size: int,
                         max_len: int,
                         training: bool = False) -> FloatTensor:
        flatten = tf.reshape(tensor, (batch_size * max_len, 1))
        sentences = self.rewnet(flatten, training=training)
        return tf.reshape(sentences, (batch_size, max_len,
                                      sentences.shape[1]))

    # ! experimental_compile doesn't work and recvec already do it
    def _no_model_inference(self,
                            tensor: Tensor,
                            batch_size: int,
                            max_len: int,
                            ce_size: int,
                            training: bool = False) -> FloatTensor:
        flatten = tf.reshape(tensor, (batch_size * max_len, 1))
        sentences = self.ce(flatten)
        ce = tf.reshape(sentences, (batch_size, max_len, ce_size))

        if self.projection_dim:
            ce = self.proj(ce)

        # FIXME for hypertuning
        if self.norm_type:
            ce = self.norm(ce, training=training)

        if self.dropout_rate:
            ce = self.dropout(ce, training=training)

        if self.spatial_dropout_rate:
            ce = self.spatial_drop(ce, training=training)

        return ce

    def _load(self, path) -> tf.keras.models.Model:
        """load rewnet model
        FIXME: handle url and name
        """
        model = tf.keras.models.load_model(path)
        model.trainable = self.trainable
        model.compile('adam', 'mse')
        return model

    def get_config(self) -> Dict[str, Any]:
        config = {
            'model': self.model,
            'sep': self.sep,
            'max_len': self.max_len,
            'eager': self.eager,
            'trainable': self.trainable,
            'verbose': self.verbose,
            'projection_dim': self.projection_dim,
            'gaussian_scale': self.gaussian_scale,
            'activations': self.activations,
            'circle_projection': self.circle_projection,
            'merge': self.merge,
            'dropout_rate': self.dropout_rate,
            'spatial_dropout_rate': self.spatial_dropout_rate,
            'norm_type': self.norm_type
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

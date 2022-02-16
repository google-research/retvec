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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="retvec")
class FourierFeatureProjection(layers.Layer):

    def __init__(self,
                 gaussian_projection_dim: int,
                 gaussian_scale: float = 10.0,
                 activations: List = ['sin', 'softsign'],
                 circle_projection: bool = True,
                 merge: str = "concat",
                 **kwargs):
        """
        Fourier Feature Projection layer

        Original idea from:
        [Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)

        Add this layer immediately after the input layer.

        Args:
            gaussian_projection_dim: Gaussian kernel Projected dimension.
                Output layer size is dim*len(activations) if concatenate,
                dim if added.

            gaussian_scale: Scale of the gaussian kernel in fourier feature
            projection layer.

            Note: If the scale is too small, convergence will slow down and
            obtain poor results. If the scale is too large (>50), convergence
            will be fast but results will be grainy.
            Try grid search for scales in the range [10 - 50].

            activations: List of functions to apply on the gaussian projection.

            circle_projection: Whether to use circle projection.

            merge: Merge strategy of projections, one of ['concat', 'add',
                or 'mul'].
        """
        super().__init__(**kwargs)

        if 'dtype' in kwargs:
            self._kernel_dtype = kwargs['dtype']
        else:
            self._kernel_dtype = None

        self.circle_projection = circle_projection
        self.activations = activations
        self.merge = merge

        self.gaussian_projection_dim = gaussian_projection_dim
        self.gauss_scale = float(gaussian_scale)
        self.pi = tf.constant(np.pi, dtype='float')

        init = tf.keras.initializers.TruncatedNormal(mean=0.0,
                                                     stddev=self.gauss_scale,
                                                     seed=3713)

        self.proj_kernel = layers.Dense(self.gaussian_projection_dim,
                                        use_bias=False,
                                        trainable=False,
                                        kernel_initializer=init,
                                        dtype=self._kernel_dtype)

    def call(self, inputs, **kwargs):
        x = inputs

        if self.circle_projection:
            x = 2.0 * self.pi * x

        # Gaussian features projection
        x = self.proj_kernel(x)

        # Smooth projection over -1 and 1 using activations
        projections = []
        for activation in self.activations:
            projections.append(self._apply_activation(x, activation))

        if self.merge == "concat":
            output = layers.Concatenate()(projections)
        elif self.merge == "add":
            output = layers.Add()(projections)
        elif self.merge == "mul":
            output = layers.Multiply()(projections)
        else:
            raise ValueError("Merge must be concat, add or mul")

        return output

    @property
    def embedding_size(self):
        if self.merge == "concat":
            return self.gaussian_projection_dim * len(self.activations)
        else:
            return self.gaussian_projection_dim

    def _apply_activation(self, x, activation):
        if activation == "sin":
            x = tf.sin(x)
        elif activation == "cos":
            x = tf.cos(x)
        elif activation:
            x = layers.Activation(activation)(x)
        return x

    def get_config(self):
        config = {
            'gaussian_projection_dim': self.gaussian_projection_dim,
            'gaussian_scale': self.gauss_scale,
            'circle_projection': self.circle_projection,
            'activations': self.activations,
            'merge': self.merge,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

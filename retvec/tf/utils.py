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

import os
from pathlib import Path
from typing import Optional

import tensorflow as tf

RETVEC_MODEL_URLS = {
    "retvec-v1": "https://storage.googleapis.com/tensorflow/keras-applications/retvec-v1"
}


def tf_cap_memory():
    """Avoid TF to hog memory before needing it"""
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def clone_initializer(initializer: tf.keras.initializers.Initializer):
    if isinstance(
        initializer,
        tf.keras.initializers.Initializer,
    ):
        return initializer.__class__.from_config(initializer.get_config())
    return initializer


def download_retvec_saved_model(
    model_name: str = "retvec-v1",
    cache_dir: str = "~/.keras/",
    model_cache_subdir: str = "retvec-v1",
):
    if model_name not in RETVEC_MODEL_URLS:
        raise ValueError(f"{model_name} is not a valid RETVec model name.")

    model_url = RETVEC_MODEL_URLS[model_name]
    model_cache_subdir_variables = f"{model_cache_subdir}/variables"

    model_components = [
        "fingerprint.pb",
        "keras_metadata.pb",
        "saved_model.pb",
    ]

    model_components_variables = [
        "variables.data-00000-of-00001",
        "variables.index",
    ]

    # download model components
    for component_name in model_components:
        tf.keras.utils.get_file(
            origin=f"{model_url}/{component_name}",
            extract=True,
            cache_subdir=model_cache_subdir,
        )

    # download variables which are under a different folder
    for component_name in model_components_variables:
        tf.keras.utils.get_file(
            origin=f"{model_url}/variables/{component_name}",
            extract=True,
            cache_subdir=model_cache_subdir_variables,
        )

    retvec_model_dir = cache_dir + model_cache_subdir
    return Path(retvec_model_dir).expanduser()

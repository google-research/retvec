"""
 Copyright 2023 Google LLC

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

# TODO (marinazh): we should download RETVec model weights instead of SavedModel files
RETVEC_COMPONENTS_HASHES = {
    "retvec-v1": {
        "fingerprint.pb": "5c3991599c293ba653c55e8cceae8e10815eeedea6aff75a64905cd71587d4c1",
        "keras_metadata.pb": "e87e8b660ef66f8a058c4c0aa8bfaa8b683bcd4669c21e4bf71055148f8c6afc",
        "saved_model.pb": "337c8e91c92946513d127b256f2872a497545186c4d2c2c09afc7d76b55454b7",
        "variables.data-00000-of-00001": "22d4760b452fe8110ef2fa96b3d84186372f5259b8f6c4041a05c3ab58d93d37",
        "variables.index": "431d19b7426b939c9834bb7d55d515a4ee7d7a6cda78ef0bf7b8ba03e67e480b",
    }
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

    # download model components
    retvec_components = RETVEC_COMPONENTS_HASHES[model_name]
    for component_name in retvec_components.keys():
        if "variables" in component_name:
            origin = f"{model_url}/variables/{component_name}"
            cache_subdir = model_cache_subdir_variables
        else:
            origin = f"{model_url}/{component_name}"
            cache_subdir = model_cache_subdir

        tf.keras.utils.get_file(
            origin=origin,
            extract=True,
            cache_subdir=cache_subdir,
            file_hash=retvec_components[component_name],
        )

    retvec_model_dir = cache_dir + model_cache_subdir
    return Path(retvec_model_dir).expanduser()

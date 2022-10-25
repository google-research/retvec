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

import pytest
import tensorflow as tf
from retvec import RetVec
from retvec.rewnet import REWCNN, REWformer
from tensorflow_similarity.losses import MultiSimilarityLoss

tf.config.set_visible_devices([], "GPU")

architectures = [REWCNN, REWformer]
architectures_names = ["rewcnn", "rewformer"]


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_basic_load(NN):
    model = NN()
    assert isinstance(model, tf.keras.Model)


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_similarity(NN):
    dim = 5
    model = NN(similarity_dim=dim)

    lyr = model.get_layer("similarity")
    assert lyr.output.shape[1] == dim


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_ori_decoder(NN):
    size = 512
    model = NN(original_decoder_size=size)

    lyr = model.get_layer("ori_decoder")
    assert lyr.output.shape[1] == size


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_aug_decoder(NN):
    size = 512
    model = NN(aug_decoder_size=size)

    lyr = model.get_layer("aug_decoder")
    assert lyr.output.shape[1] == size


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_aug_vector(NN):
    dim = 4
    model = NN(max_chars=16, aug_vector_dim=dim)

    lyr = model.get_layer("aug_vector")

    assert lyr.output.shape[1] == 16
    assert lyr.output.shape[2] == dim


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_aug_matrix(NN):
    dim = 4
    model = NN(max_chars=16, aug_matrix_dim=dim)

    lyr = model.get_layer("aug_matrix")

    assert lyr.output.shape[1] == 16
    assert lyr.output.shape[2] == dim


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_save_and_reload(tmpdir, NN):

    path = str(tmpdir / "test/")
    model = NN()
    model.compile(optimizer="adam", loss=MultiSimilarityLoss())
    model.save(path)
    reloaded_model = tf.keras.models.load_model(path)
    reloaded_model.summary()


@pytest.mark.parametrize("NN", architectures, ids=architectures_names)
def test_extract_tokenizer(tmpdir, NN):
    path = str(tmpdir / "test/")
    model = NN()
    tokenizer = tf.keras.Model(model.input, model.get_layer("tokenizer").output)
    tokenizer.compile("adam", "mse")
    tokenizer.save(path)
    _ = RetVec(model=path)

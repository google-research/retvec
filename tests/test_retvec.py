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

import tensorflow as tf
from retvec import RetVec
from tensorflow.keras import layers

tf.config.set_visible_devices([], 'GPU')

char_embedding_size = 16


def test_eager_mode():
    max_len = 12
    batch = ["hello world", "this is a batch"]
    tok = RetVec(model=None, max_len=max_len, eager=True)
    vals = tok(batch)
    assert vals.shape == (2, max_len, 16 * char_embedding_size)


def test_graph_mode():
    max_len = 12
    batch = ["hello world", "this is a batch"]

    i = layers.Input(shape=(1), dtype='string')
    x = i
    x = RetVec(model=None, max_len=max_len)(x)
    mdl = tf.keras.Model(i, x)
    mdl.summary()
    preds = mdl.predict(batch)
    assert preds.shape == (2, max_len, 16 * char_embedding_size)


def test_model_reload(tmp_path):
    max_len = 12
    i = layers.Input(shape=(1), dtype='string')
    x = i
    x = RetVec(model=None, max_len=max_len)(x)
    mdl = tf.keras.Model(i, x)
    mdl.compile('rmsprop', 'mse')
    save_path = tmp_path / "retvec_test_model/"
    mdl.save(save_path)
    tf.keras.models.load_model(save_path)


def test_model_reload_project(tmp_path):
    max_len = 12
    i = layers.Input(shape=(1), dtype='string')
    x = i
    x = RetVec(model=None,
               max_len=max_len,
               projection_dim=256,
               dropout_rate=0.05,
               spatial_dropout_rate=0.05)(x)
    mdl = tf.keras.Model(i, x)
    mdl.compile('rmsprop', 'mse')
    save_path = tmp_path / "retvec_test_model/"
    mdl.save(save_path)
    tf.keras.models.load_model(save_path)


def test_determinism():
    batch = ["hello world 😀", "hello world 😀"]

    tok = RetVec(model=None, eager=True)
    vects = tok(batch)
    vects2 = tok(batch)
    assert tf.reduce_all(tf.equal(vects[0], vects[1]))
    assert tf.reduce_all(tf.equal(vects[0], vects2[1]))


def test_base_parameters():
    max_len = 8
    batch = ["hello-world-😀-diff", "hello-😀-world-diff2"]

    tok = RetVec(model=None, sep='-', max_len=max_len, eager=True)
    vects = tok(batch)
    assert vects.shape == (2, max_len, 16 * char_embedding_size)
    assert tf.reduce_sum(vects[0][2]) == tf.reduce_sum(vects[1][1])  # 😀
    assert tf.reduce_sum(vects[0][3]) != tf.reduce_sum(vects[1][3])  # diff - diff2

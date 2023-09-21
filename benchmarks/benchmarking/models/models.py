
from typing import Dict
import tensorflow as tf
from tensorflow.keras.layers import Layer

from . import bert
from . import rnn
from . import dpcnn
from . import dummy

MODEL_FNS = {
    'bert': bert.from_config,
    'rnn': rnn.from_config,
    'dpcnn': dpcnn.from_config,
    'dummy': dummy.from_config
}


def get_model(vectorizer: Layer,
              config: Dict) -> tf.keras.Model:
    model_fn = MODEL_FNS[config['model']['type']]
    return model_fn(vectorizer, config)

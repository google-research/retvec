# RETVec: Resilient & Efficient Text Vectorizer


## Overview
RETVec is a next-gen text vectorizer designed to be efficient, multilingual, and provide built-in adversarial resilience using robust word embeddings trained with [similarity learning](https://github.com/tensorflow/similarity/). You can read the paper [here](https://arxiv.org/abs/2302.09207).

RETVec is trained to be resilient against character-level manipulations including insertion, deletion, typos, homoglyphs, LEET substitution, and more. The RETVec model is trained on top of a novel character encoder which can encode all UTF-8 characters and words efficiently. Thus, RETVec works out-of-the-box on over 100 languages without the need for a lookup table or fixed vocabulary size. Furthermore, RETVec is a layer, which means that it can be inserted into any TF model without the need for a separate pre-processing step.

RETVec's speed and size (~200k instead of millions) also makes it a great choice for on-device and web use cases. It is natively supported in TensorFlow Lite via custom ops implemented in TensorFlow Text, and we provide a Javascript implementation of RETVec which allows you to deploy web models via TensorFlow.js.

Please see our example colabs on how to get started with training your own models with RETVec.

## Getting started


### Installation

You can use pip to install the latest TensorFlow version of RETVec:

```python
pip install retvec
```

RETVec has been tested on TensorFlow 2.6+ and python 3.7+.

### Basic Usage

You can use RETVec as the vectorization layer in any TensorFlow model with just a single line of code. RETVec operates on raw strings with pre-processing options built-in (e.g. lowercasing text). For example:

```
import tensorflow as tf
from tensorflow.keras import layers

# Define the input layer, which accepts raw strings
inputs = layers.Input(shape=(1, ), name="input", dtype=tf.string)

# Add the RETVec Tokenizer layer using the RETVec embedding model -- that's it!
x = RETVecTokenizer(sequence_length=128)(inputs)

# Create your model like normal
# e.g. a simple LSTM model for classification with NUM_CLASSES classes
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```

Then you can compile, train and save your model like usual! As demonstrated in our paper, models trained using RETVec are more resilient against adversarial attacks and typos, as well as computationally efficient. RETVec also offers support in TFJS and TF Lite, making it perfect for on-device mobile and web use cases.

### Colabs

Detailed example colabs for RETVec can be found at under [notebooks](notebooks/). These are a good way to get started with using RETVec. You can run the notebooks in Google Colab by clicking the Google Colab button. If none of the examples are similar to your use case, please let us know!

We have the following example colabs:

- Training RETVec-based models using TensorFlow: [train_hello_world_tf.ipynb](notebooks/train_hello_world_tf.ipynb) for GPU/CPU training, and [train_tpu.ipynb](notebooks/train_tpu.ipynb) for a TPU-compatible training example.
- (Coming soon!) Converting RETVec models into TF Lite models to run on-device.
- (Coming soon!) Using RETVec JS to deploy RETVec models in the web using TensorFlow.js

## Citing
Please cite this reference if you use RETVec in your research:

```bibtex
@article{retvec2023,
    title={RETVec: Resilient and Efficient Text Vectorizer}, 
    author={Elie Bursztein, Marina Zhang, Owen Vallis, Xinyu Jia, and Alexey Kurakin},
    year={2023},
    eprint={2302.09207}
}
```

## Contributing
To contribute to the project, please check out the [contribution guidelines](CONTRIBUTING.md). Thank you!

## Disclaimer
This is not an official Google product.

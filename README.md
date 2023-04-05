# RETVec: Resilient & Efficient Text Vectorizer


## Overview
RETVec is a next-gen text vectorizer designed to offer built-in adversarial resilience using robust word embeddings. Read the paper here: https://arxiv.org/abs/2302.09207.

RETVec is trained to be resilient against character manipulations including insertion, deletion, typos, homoglyphs, LEET substitution, and more. The RETVec model is trained on top of a novel character embedding which can encode all UTF-8 characters and words. Thus, RETVec works out-of-the-box on over 100 languages without the need for a lookup table or fixed vocabulary size. Furthermore, RETVec is a layer, which means that it can be inserted into any TF model without the need for a separate pre-processing step.


### Getting started

#### Installation

You can use pip to install the TensorFlow version of RETVec:

```python
pip install retvec
```

RETVec has been tested on TensorFlow 2.6+ and python 3.7+.

### Basic Usage

`training/train_tf_retvec_models.py` is the RETVec model training script. Example usage:

```python
train_tf_retvec_models.py --train_config <train_config_path> --model_config <model_config_path> --output_dir <output_path>
```

Configurations for our base models are under the `configs/` folder.

### Colab

Colab for training and releasing a new RETVec model: `notebooks/train_and_relase_a_rewnet.ipynb`

Hello world colab: `notebooks/hello_world.ipynb`

## Disclaimer
This is not an official Google product.

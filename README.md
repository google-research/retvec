# RetVect: Resilient & Efficient Text Vectorizer


## Overview
RetVec is a next-gen text vectorizer designed to offer built-in adversarial resilience using robust word embeddings.

RetVec is trained to be resilient against character manipulations including insertion, deletion, typos, homoglyphs, LEET substitution, and more. The RetVec model is trained on top of a novel character embedding which can encode all UTF-8 characters and words. Thus, RetVec works out-of-the-box on over 100 languages without the need for a lookup table or fixed vocabulary size. Furthermore, RetVec is a layer, which means that it can be inserted into any TF model without the need for a separate pre-processing step.


### Getting started

#### Installation

```python
pip install retvec
```

RetVec has been tested on TensorFlow 2.6+ and python 3.7+.

### Basic Usage

`train_rewnet.py` is the REWNet training script. Example usage:

```python
cli/train_rewnet.py --train_config <train_config_path> --model_config <model_config_path> --output_dir <output_path>
```

Configurations for our base models are under the `configs/` folder.

### Colab

Colab for training and releasing a new REWNet model: `notebooks/train_and_relase_a_rewnet.ipynb`

Hello world colab: `notebooks/hello_world.ipynb`

## Disclaimer
This is not an official Google product.

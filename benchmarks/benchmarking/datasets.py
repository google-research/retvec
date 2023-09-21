from typing import Dict, List, Optional, Tuple, Union, no_type_check
import re

from tqdm.auto import tqdm
# import stanza
import tensorflow as tf
import numpy as np
from tensorflow import Tensor
import fasttext.util
import random

from datasets import load_dataset, load_from_disk


class Datasets():
    """Class that helps simplify the process of loading datasets from
        huggingface datasets.

        For more info on huggingface datasets:
        # https://huggingface.co/datasets?search=news
        # https://huggingface.co/datasets/viewer/
    """

    def __init__(self) -> None:
        self.db = {
            'ag_news': {
                'path': 'ag_news',
                'language': 'en',
                'num_labels': 4,
                'text': 'text',
                'labels': 'label',
                'train': 'train',
                'test': 'test'
            },
            'amazon_reviews_de': {
                'path': 'amazon_reviews_multi',
                'name': 'de',
                'language': 'de',
                'num_labels': 5,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test'
            },
            'amazon_reviews_polarity_de': {
                'path': 'amazon_reviews_multi',
                'name': 'de',
                'language': 'de',
                'num_labels': 2,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'preprocess_fns': [self._preprocess_reviews_polarity]
            },
            'amazon_reviews_en': {
                'path': 'amazon_reviews_multi',
                'name': 'en',
                'language': 'en',
                'num_labels': 5,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test'
            },
            'amazon_reviews_polarity_en': {
                'path': 'amazon_reviews_multi',
                'name': 'en',
                'language': 'en',
                'num_labels': 2,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'preprocess_fns': [self._preprocess_reviews_polarity]
            },
            'amazon_reviews_fr': {
                'path': 'amazon_reviews_multi',
                'name': 'fr',
                'language': 'fr',
                'num_labels': 5,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test'
            },
            'amazon_reviews_polarity_fr': {
                'path': 'amazon_reviews_multi',
                'name': 'fr',
                'language': 'fr',
                'num_labels': 2,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'preprocess_fns': [self._preprocess_reviews_polarity]
            },
            'amazon_reviews_es': {
                'path': 'amazon_reviews_multi',
                'name': 'es',
                'language': 'es',
                'num_labels': 5,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test'
            },
            'amazon_reviews_polarity_es': {
                'path': 'amazon_reviews_multi',
                'name': 'es',
                'language': 'es',
                'num_labels': 2,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'preprocess_fns': [self._preprocess_reviews_polarity]
            },
            'amazon_reviews_zh': {
                'path': './datasets/huggingface/amazon_reviews_multi_zh_stanza_words',
                'language': 'zh',
                'num_labels': 5,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'load_from_disk': True
            },
            'amazon_reviews_polarity_zh': {
                'path': './datasets/huggingface/amazon_reviews_multi_zh_stanza_words',
                'language': 'zh',
                'num_labels': 2,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'preprocess_fns': [self._preprocess_reviews_polarity],
                'load_from_disk': True
            },
            'amazon_reviews_ja': {
                'path': './datasets/huggingface/amazon_reviews_multi_ja_stanza_words',
                'language': 'ja',
                'num_labels': 5,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'load_from_disk': True
            },
            'amazon_reviews_polarity_ja': {
                'path': './datasets/huggingface/amazon_reviews_multi_ja_stanza_words',
                'language': 'ja',
                'num_labels': 2,
                'text': ['review_title', 'review_body'],
                'labels': 'stars',
                'train': 'train',
                'test': 'test',
                'preprocess_fns': [self._preprocess_reviews_polarity],
                'load_from_disk': True
            },
            'sst2': {
                'path': 'glue',
                'name': 'sst2',
                'language': 'en',
                'num_labels': 2,
                'text': 'sentence',
                'labels': 'label',
                'train': 'train',
                'test': 'validation'
            },
            'cola': {
                'path': 'glue',
                'name': 'cola',
                'language': 'en',
                'num_labels': 2,
                'text': 'sentence',
                'labels': 'label',
                'train': 'train',
                'test': 'validation'
            },
            'rte': {
                'path': 'glue',
                'name': 'rte',
                'language': 'en',
                'num_labels': 2,
                'text': ['sentence1', 'sentence2'],
                'labels': 'label',
                'train': 'train',
                'test': 'validation'
            },
            'yelp_polarity': {
                'path': 'yelp_polarity',
                'language': 'en',
                'num_labels': 2,
                'text': 'text',
                'labels': 'label',
                'train': 'train',
                'test': 'test'
            },
            'rotten_tomatoes': {
                'path': 'rotten_tomatoes',
                'language': 'en',
                'num_labels': 2,
                'text': 'text',
                'labels': 'label',
                'train': 'train',
                'test': 'test'
            },
            'urdu_fake_news': {
                'path': 'urdu_fake_news',
                'language': 'ur',
                'num_labels': 2,
                'text': 'news',
                'labels': 'label',
                'train': 'train',
                'test': 'test'
            },
            'imdb': {
                'path': 'imdb',
                'language': 'en',
                'num_labels': 2,
                'text': 'text',
                'labels': 'label',
                'train': 'train',
                'test': 'test'
            },
            'dbpedia_14': {
                'path': 'dbpedia_14',
                'language': 'en',
                'num_labels': 14,
                'text': ['title', 'content'],
                'labels': 'label',
                'train': 'train',
                'test': 'test'
            },
            'yahoo_answers_topics': {
                'path': 'yahoo_answers_topics',
                'language': 'en',
                'num_labels': 10,
                'text': ['question_title', 'question_content', 'best_answer'],
                'labels': 'topic',
                'train': 'train',
                'test': 'test'
            },
            'massive_intent': {
                'path': 'qanastek/MASSIVE',
                'name': 'all',
                'language': 'multi',
                'num_labels': 60,
                'text': ['utt'],
                'labels': 'intent',
                'train': 'train',
                'test': 'test'
            }
        }

    @no_type_check
    def get(self,
            dataset_name: Union[List[str], str],
            datasets: List[str] = None,
            data_dir: Optional[str] = None,
            split: Optional[str] = None,
            one_hot: bool = True,
            config: Dict = None,
            use_tfds: bool = True,
            tds_shuffle_buffer: int = 100000) -> Tuple[Tensor, Tensor]:
        """Return the dataset requested as tensors. All arguments
        are consistent with datasets.load_dataset()
        https://huggingface.co/docs/datasets/loading_datasets.html.

        Args:
            dataset_name: Dataset name, must be a key in self.db.
            datasets: Instead of a single dataset name, 'datasets' will be
                a list of dataset names that appear as keys in self.db,
                in which the datasets will be returned as one dataset, i.e.
                ['amazon_reviews_en', 'amazon_reviews_fr'] will return both
                the en and fr splits of the Amazon reviews dataset.
            split: Defines which split of the data to load.
            data_dir: Defining the data_dir of the dataset configuration,
                only necessary for datasets that need additional manually
                downloaded files, i.e. jigsaw_toxicity_pred.
            one_hot: One hot labels.
        Raises:
            ValueError: If the dataset is not in the database.

        Returns:
            The text and label tensors corresponding to the requested dataset,
            with the labels one-hot encoded.
        """
        texts = []
        labels = []
        langs = []

        # single dataset requested
        if dataset_name and not datasets:
             return self._load_dataset(
                dataset_name, data_dir, split, one_hot, config)

        # multiple datasets requested, return as one (combined) dataset
        else:
            for ds_name in datasets:
                x, y, l = self._load_dataset(
                    ds_name, data_dir, split, one_hot, config)
                texts.append(x)
                labels.append(y)
                langs.append(l)

            # concatenate all datasets together
            if isinstance(texts[0], np.ndarray):
                texts = np.concatenate(texts, axis=0)
                labels = np.concatenate(labels, axis=0)
            else:
                texts = tf.concat(texts, axis=0)
                labels = tf.concat(labels, axis=0)

            langs = tf.concat(langs, axis=0)

            if config['vectorizer']['type'] == 'fasttext':
                texts = tf.constant(texts)
                labels = tf.constant(labels)
                indices = tf.range(start=0, limit=tf.shape(texts)[0], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices)
                shuffled_texts = tf.gather(texts, shuffled_indices)
                shuffled_labels = tf.gather(labels, shuffled_indices)
                shuffled_langs = tf.gather(langs, shuffled_indices)
                return shuffled_texts, shuffled_labels, langs
            else:
                return texts, labels, langs

    @no_type_check
    def _load_dataset(self,
                      dataset_name,
                      data_dir: Optional[str] = None,
                      split: Optional[str] = 'train',
                      one_hot: bool = True,
                      config: Dict = None) -> Tuple[Tensor, Tensor]:
        if dataset_name not in self.db:
            raise ValueError('Unknown dataset - need to add it')
        info = self.db[dataset_name]

        if data_dir is None:
            data_dir = info.get('data_dir', None)  # type: Optional[str]

        # if dataset path doesn't exist in huggingface then try loading from disk
        if info.get('load_from_disk', False):
            dataset = load_from_disk(info.get('path'))
            dataset = dataset[split]
        else:
            dataset = load_dataset(info.get('path'),
                                   info.get('name'),
                                   split=split,
                                   data_dir=data_dir)

        # shuffle dataset here
        dataset = dataset.shuffle()

        # concatenate text if there are multiple fields, i.e. review title and
        # body from the amazon reviews dataset
        with tf.device('/cpu:0'):

            # if language is chinese or japanese for amazon, we use stanza first to split into words and pad punctuation
            if 'amazon' in info.get('path') and info['language'] in ['zh', 'ja']:    
                text_field_suffix = '_stanza_words'
            else:
                text_field_suffix = ''

            if isinstance(info['text'], list):
                x = []
                for field in info['text']:
                    x.append(tf.constant(dataset[field + text_field_suffix], dtype=tf.string))
                x = tf.strings.join(x, separator=f" ")
            else:
                x = tf.constant(dataset[info['text'] + text_field_suffix], dtype=tf.string)

            y = tf.constant(dataset[info['labels']], dtype=tf.int32)

            # pad puncutation
            if info.get('language', None) not in ['zh', 'ja']:
                x, y = self._pad_punctuation(x, y)

            if not config or config["lowercase"]:
                x, y = self._lowercase(x, y)

            # clean whitespace
            x, y = self._collapse_whitespace(x, y)

            # other preprocessing functions
            if info.get('preprocess_fns'):
                for pp_fn in info['preprocess_fns']:
                    x, y = pp_fn(x, y)

            # insert cls token at the beginning for BERT models
            # BPE vectorizer and sentencepiece insert cls token later
            if config and config['model']['type'] == "bert":
                if config['vectorizer']['type'] not in ['byte_level_bpe', 'sentencepiece']:
                    x, y = self._preprocess_insert_cls(x, y, cls_token='[CLS]')

            # one hot labels
            if one_hot:
                y = tf.one_hot(y, depth=info['num_labels'])

        lang = info['language']
        langs = [lang for _ in range(x.shape[0])]
        langs = tf.constant(langs, dtype=tf.string)
        return x, y, langs

    def _preprocess_reviews_polarity(self, texts, labels):
        """ Converts 1 and 2 star ratings to 0 'negative'
            and 4 and 5 star ratings to 1 'positive',
            ignores 3 star reviews.
        """
        texts = texts.numpy()
        labels = labels.numpy()
        text_polarity = []
        labels_polarity = []

        # convert to numpy because too slow to iterate over tensors
        for i in range(len(texts)):
            if labels[i] <= 2:
                labels_polarity.append(0)
                text_polarity.append(texts[i])
            elif labels[i] >= 4:
                labels_polarity.append(1)
                text_polarity.append(texts[i])

        text_polarity = tf.constant(text_polarity, tf.string)
        labels_polarity = tf.constant(labels_polarity, tf.int32)
        return text_polarity, labels_polarity

    def _preprocess_insert_cls(self, texts, labels, cls_token='[CLS]'):
        """Inserts a cls token to the beginning of each input."""
        cls_tensor = [cls_token] * len(texts)
        cls_tensor = tf.constant(cls_tensor, tf.string)
        texts_cls = tf.strings.join([cls_tensor, texts], ' ')
        return texts_cls, labels

    def _pad_punctuation(self, texts, labels):

        def _pad_punctuation_single_text(text: str) -> str:
            """Adds spaces around punctuation. Not hyphens and apostrophies, because
            those are typically inside of words."""

            # ! If you want to ignore URLs: (https?://\S*)|[[:punct:]]+
            return re.sub(r'([!"#$%&()*+,./:;<=>?@[\\\]^_`{|}~])', r' \1 ', text)

        texts = texts.numpy()
        texts_padded = []

        # convert to numpy because too slow to iterate over tensors
        for text in texts:
            texts_padded.append(_pad_punctuation_single_text(text.decode("utf-8")))

        texts_padded = tf.constant(texts_padded, tf.string)
        return texts_padded, labels

    def _remove_punctuation(self, texts, labels):

        def _remove_punctuation_single_text(text: str) -> str:
            """Adds spaces around punctuation. Not hyphens and apostrophies, because
            those are typically inside of words."""

            # ! If you want to ignore URLs: (https?://\S*)|[[:punct:]]+
            return re.sub(r'([!"#$%&()*+,./:;<=>?@[\\\]^_`{|}~])', r' ', text)

        texts = texts.numpy()
        texts_padded = []

        # convert to numpy because too slow to iterate over tensors
        for text in texts:
            texts_padded.append(_remove_punctuation_single_text(text.decode("utf-8")))

        texts_padded = tf.constant(texts_padded, tf.string)
        return texts_padded, labels

    def _collapse_whitespace(self, texts, labels):

        def _collapse_whitespace_single_text(text: str) -> str:
            """Adds spaces around punctuation. Not hyphens and apostrophies, because
            those are typically inside of words."""
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text

        texts = texts.numpy()
        texts_padded = []

        # convert to numpy because too slow to iterate over tensors
        for text in texts:
            texts_padded.append(_collapse_whitespace_single_text(text.decode("utf-8")))

        texts_padded = tf.constant(texts_padded, tf.string)
        return texts_padded, labels

    def _lowercase(self, texts, labels):
        texts = texts.numpy()
        texts_padded = []

        # convert to numpy because too slow to iterate over tensors
        for text in texts:
            texts_padded.append(text.decode("utf-8").lower())

        texts_padded = tf.constant(texts_padded, tf.string)
        return texts_padded, labels

import fasttext
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from time import time


def get_fasttext_dict():
    print('load fasttext dict')
    preprocess_start = time()

    fasttext_langs = ["af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv", "ml", "mn", "ms", "my", "no", "nl", "pl", "pt", "ro", "ru", "sl", "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "zh"]
    
    fasttext_bin_map = {}

    for lang in tqdm(fasttext_langs, 'loading fasttext'):
        ft = fasttext.load_model(f"./benchmarking/data/fasttext_bin/cc.{lang}.300.bin")
        fasttext_bin_map[lang] = ft
    preprocess_time = time() - preprocess_start
    print('load fasttext dict time', preprocess_time)
    return fasttext_bin_map


FASTTEXT_BIN_MAP = get_fasttext_dict()

def get_fasttext_vector(text, lang):
    text = text.numpy().decode()
    lang = lang.numpy().decode()

    words = text.split()
    # tf.print(words)
    words = words[:128]
    text_emb = np.zeros((128, 300))
    for i in range(len(words)):
        word = words[i]
        word_emb = FASTTEXT_BIN_MAP[lang].get_word_vector(word)
        text_emb[i] = np.array(word_emb)
    return tf.constant(text_emb, dtype=tf.float32)

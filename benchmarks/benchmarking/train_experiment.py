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
from time import time
import json
import tensorflow as tf
import wandb

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from wandb.keras import WandbCallback

from typing import Dict
from .datasets import Datasets
from .vectorizers import get_vectorizer
from .preprocessors import preprocess_text
from .models import get_model
from .optimizers import create_optimizer

class TrainExperiment:

    # dataset loader
    datasets = Datasets()

    def __init__(self, config: Dict) -> None:
        self.config = config

    def run(self) -> None:
        config = self.config

        # set up save directories
        base_dir = config['base_dir']
        
        print(config)

        dataset_name = config['dataset']['name']
        model_name = config['model']['name']
        vec_name = config['vectorizer']['name']

        exp_name = f"{dataset_name}_{model_name}_{vec_name}_{int(time())}"
        results_dir = os.path.join(base_dir, 'results', exp_name)
        model_dir = os.path.join(base_dir, 'models', exp_name)
        log_dir = os.path.join(base_dir, 'logs', exp_name)

        # create dirs if they don't exist
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        if config["wandb_project"]:
            wandb.init(
                project=config["wandb_project"],
                entity="marinazh",
                group=config['dataset']['name'],
                name=f"{model_name}_{vec_name}"
            )
            wandb.config = config

        print('Saving results to ' + results_dir)
        print('Saving models to ' + model_dir)
        print('Saving logs to ' + log_dir)

        # save directories in config
        config['results_dir'] = results_dir
        config['model_dir'] = model_dir
        config['log_dir'] = log_dir
        config['exp_name'] = exp_name

        # load train and test dataset
        dataset_name = config['dataset']['name']

        # a list of dataset names, if neccessary
        # i.e. for amazon reviews all languages, which is a combination
        # of multiple datasets
        datasets = config['dataset'].get('datasets', None)

        if config['vectorizer']['type'] == 'fasttext':
            x_train, y_train, lang_train = TrainExperiment.datasets.get(
                dataset_name, datasets=datasets, split='train', config=config)
            x_test, y_test, lang_test = TrainExperiment.datasets.get(
                dataset_name, datasets=datasets, split='test', config=config)

        else:
            x_train, y_train, _ = TrainExperiment.datasets.get(
                dataset_name, datasets=datasets, split='train', config=config)
            x_test, y_test, _ = TrainExperiment.datasets.get(
                dataset_name, datasets=datasets, split='test', config=config)

        # truncate dataset to fit batch if steps_per_epoch not set
        batch_size = config['batch_size']

        # get vectorizer and tokenizer
        vectorizer, tokenizer = get_vectorizer(x_train, config)

        # preprocess
        x_train = preprocess_text(x_train, vectorizer, tokenizer, config)
        x_test = preprocess_text(x_test, vectorizer, tokenizer, config)

        # model
        model = get_model(vectorizer, config)
        model.summary()

        callbacks = [TensorBoard(log_dir=log_dir),
                     ModelCheckpoint(model_dir, monitor='val_acc', save_best_only=True)]

        if config["wandb_project"]:
            callbacks.append(WandbCallback(save_model=False))

        if config["early_stopping_patience_steps"]:
            early_stop_epochs = config["early_stopping_patience_steps"] // config["steps_per_epoch"]
            callbacks.append(EarlyStopping(monitor="val_acc", min_delta=0, patience=early_stop_epochs))

        optimizer = create_optimizer(init_lr=config["optimizer"]["init_lr"],
                                     num_train_steps=config["num_train_steps"],
                                     num_warmup_steps=config.get("num_warmup_steps", 0),
                                     weight_decay_rate=config["optimizer"].get("weight_decay_rate", 0.0),
                                     optimizer_type=config["optimizer"].get("type", "adam"),
                                     beta_1=config["optimizer"].get("beta_1", 0.9),
                                     beta_2=config["optimizer"].get("beta_2", 0.999),
                                     end_lr=config["optimizer"].get("end_lr", 0.0),
                                     decay_fn=config["optimizer"].get("decay_fn", "cosine"))

        metrics = config['metrics']

        if not isinstance(metrics, list):
            metrics = [metrics]

        model.compile(optimizer,
                      loss=config['loss'],
                      metrics=config['metrics'])

        epochs = config["num_train_steps"] // config["steps_per_epoch"]

        shuffle_buffer_size = 10000 if config['vectorizer']['type'] == 'fasttext' else x_train.shape[0]

        x_train_ds = tf.data.Dataset.from_tensor_slices(x_train)

        if config['vectorizer']['type'] == 'fasttext':
            from .fasttext_utils import get_fasttext_vector

            lang_train_ds = tf.data.Dataset.from_tensor_slices(lang_train)
            x_train_ds = tf.data.Dataset.zip((x_train_ds, lang_train_ds))
            x_train_ds = x_train_ds.map(lambda x, y: tf.py_function(func=get_fasttext_vector, inp=[x, y], Tout=tf.float32))

        y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)
        train_ds = tf.data.Dataset.zip((x_train_ds, y_train_ds))
        train_ds = train_ds.shuffle(shuffle_buffer_size)
        train_ds = train_ds.repeat()
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(10)

        x_test_ds = tf.data.Dataset.from_tensor_slices(x_test)

        if config['vectorizer']['type'] == 'fasttext':
            lang_test_ds = tf.data.Dataset.from_tensor_slices(lang_test)
            x_test_ds = tf.data.Dataset.zip((x_test_ds, lang_test_ds))
            x_test_ds = x_test_ds.map(lambda x, y: tf.py_function(func=get_fasttext_vector, inp=[x, y], Tout=tf.float32))

        y_test_ds = tf.data.Dataset.from_tensor_slices(y_test)
        test_ds = tf.data.Dataset.zip((x_test_ds, y_test_ds))
        test_ds = test_ds.batch(batch_size)

        history = model.fit(train_ds,
                            validation_data=test_ds,
                            epochs=epochs,
                            steps_per_epoch=config['steps_per_epoch'],
                            callbacks=callbacks)

        with open(results_dir + '/train_history.json', 'w') as f:
            json.dump(history.history, f)

        with open(results_dir + '/train_config.json', 'w') as f:
            json.dump(config, f)

        print("Checking saved model at %s" % model_dir)
        saved_model = tf.keras.models.load_model(model_dir)
        saved_model.summary()

        wandb.finish()

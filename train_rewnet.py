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

import argparse
import json
import os
from time import time
from typing import Dict

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow_addons.optimizers import LAMB
from termcolor import cprint
from wandb.keras import WandbCallback

from retvec.datasets.io import get_dataset_samplers, get_outputs_info
from retvec.optimizers import WarmUpCosine
from retvec.rewnet.rewcnn import build_rewcnn_from_config
from retvec.rewnet.rewformer import build_rewformer_from_config
from retvec.utils import tf_cap_memory


def train(args: argparse.Namespace, config: Dict) -> None:

    # save paths
    model_name = "%s-v%s" % (config['name'], config['version'])
    cprint('[Model: %s]' % (model_name), 'yellow')
    cprint('|-epochs: %s' % config['train']['epochs'], 'blue')
    cprint('|-steps_per_epoch: %s' %
           config['train']["steps_per_epoch"], 'green')
    cprint('|-batch_size: %s' % config['train']['batch_size'], 'blue')
    stub = "%s_%s" % (model_name, int(time()))
    mdl_path = "%s/rewnet/%s" % (args.output_dir, stub)
    log_dir = '%s/logs/%s' % (args.output_dir, stub)

    if args.wandb_project:
        wandb.init(project=args.wandb_project,
                   entity="marinazh",
                   group=config['model']['type'],
                   name=model_name)
        wandb.config = config

    # dataset
    train_ds, test_ds = get_dataset_samplers(bucket=args.dataset_bucket,
                                             train_path=args.train_dataset_path,
                                             test_path=args.test_dataset_path,
                                             config=config)

    # callbacks
    epochs = config['train']['epochs']
    steps_per_epoch = config['train']['steps_per_epoch']
    total_steps = epochs * steps_per_epoch
    save_freq_epochs = config['train']['save_freq_epochs']
    validation_steps = config['train']['validation_steps']

    if save_freq_epochs:
        save_freq = save_freq_epochs * steps_per_epoch
        mcc = ModelCheckpoint(mdl_path + '/epoch_{epoch}',
                              monitor='loss',
                              save_freq=save_freq)
    else:
        mcc = ModelCheckpoint(
            mdl_path, monitor='loss', save_best=True)

    tbc = TensorBoard(log_dir=log_dir, update_freq='epoch')
    callbacks = [tbc, mcc]

    if args.wandb_project:
        callbacks.append(WandbCallback(save_model=False))

    loss, metrics, _ = get_outputs_info(config)

    # mirrored strategy for multi gpu
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # model
    with mirrored_strategy.scope():
        model_type = config['model']['type']
        if model_type == 'rewcnn':
            model = build_rewcnn_from_config(config)
        elif model_type == 'rewformer':
            model = build_rewformer_from_config(config)
        else:
            raise ValueError('Unknown model %s' % model_type)

        lr_schedule = WarmUpCosine(
            initial_learning_rate=config['train']['init_lr'],
            decay_steps=total_steps,
            warmup_steps=config['train']['warmup_steps'],
            warmup_learning_rate=config['train']['warmup_learning_rate'],
            alpha=config['train']['end_lr'] / config['train']['init_lr'])

        if config['train']['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_schedule)

        if config['train']['optimizer'] == 'lamb':
            optimizer = LAMB(lr_schedule)

        model.summary()
        model.compile(optimizer, loss=loss, metrics=metrics)

    # train
    _ = model.fit(train_ds,
                  validation_data=test_ds,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  validation_steps=validation_steps)

    # extract and save tokenizer
    rew_path = "%s/embeddings/%s" % (args.output_dir, stub)

    # check that model can be reloaded
    if save_freq_epochs:
        saved_model = tf.keras.models.load_model(
            mdl_path + '/epoch_%s' % epochs)
    else:
        saved_model = tf.keras.models.load_model(mdl_path)

    # embedding is always the second layer after Input layer and Binarizer
    embedding_model = tf.keras.Model(saved_model.layers[2].input,
                                     saved_model.get_layer('tokenizer').output)
    embedding_model.compile('adam', 'mse')
    embedding_model.summary()
    embedding_model.save(rew_path, include_optimizer=False)
    cprint(f"Saving RetVec embedding model to {rew_path}", 'blue')

    wandb.finish()


def main(args: argparse.Namespace) -> None:
    # grow gpu memory usage when neccessary
    tf_cap_memory()

    # config is a single json file or a folder
    if str(args.model_config).endswith('.json'):
        model_config_paths = [args.model_config]

    else:
        c_dir = os.listdir(args.model_config)
        model_config_paths = [args.model_config +
                              f for f in c_dir if f.endswith('.json')]

    for model_config_path in model_config_paths:
        with open(model_config_path) as f:
            model_config = json.load(f)
        with open(args.train_config) as f:
            train_config = json.load(f)

        config = model_config
        config["train"] = train_config
        train(args, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetVec Training')
    parser.add_argument('--train_config', '-c', help='train config path',
                        default='configs/train_full.json')
    parser.add_argument('--model_config', '-m',
                        help='model config file or folder path')
    parser.add_argument('--output_dir', '-o', help='base output directory',
                        default='./experiments/')
    parser.add_argument('--dataset_bucket', '-p', help='gcs bucket of dataset',
                        default='retvec_datasets')
    parser.add_argument('--train_dataset_path',
                        help='gcs path to training dataset',
                        default='training/retvec_fasttext_deduplicated_v2/')
    parser.add_argument('--test_dataset_path',
                        help='gcs path to testing dataset',
                        default='training/retvec_fasttext_deduplicated_v2/')
    parser.add_argument('--wandb_project', default='RetVec-REWNet',
                        help='Wandb project to save to, none to disable.')
    args = parser.parse_args()

    main(args)

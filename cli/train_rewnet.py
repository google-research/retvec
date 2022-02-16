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

import tensorflow as tf
from retvec.datasets.io import get_dataset_samplers
from retvec.rewnet.rewbert import build_rewbert_from_config
from retvec.rewnet.rewcnn import build_rewcnn_from_config
from retvec.rewnet.rewmix import build_rewmix_from_config
from retvec.utils import get_outputs_info, tf_cap_memory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay, PolynomialDecay
from termcolor import cprint


def train(args, config):

    # save paths
    model_name = "%s-%s-v%s" % (args.model, config['name'], config['version'])
    cprint('[Model: %s]' % (model_name), 'yellow')
    cprint('|-epochs: %s' % config['epochs'], 'blue')
    cprint('|-steps_per_epoch: %s' % config["steps_per_epoch"], 'green')
    cprint('|-batch_size: %s' % config['batch_size'], 'blue')
    stub = "%s_%s" % (model_name, int(time()))
    mdl_path = "%s/rewnet/%s" % (args.output_dir, stub)
    log_dir = '%s/logs/%s' % (args.output_dir, stub)

    # dataset
    train_ds, test_ds = get_dataset_samplers(bucket=args.dataset_bucket,
                                             path=args.dataset_path,
                                             config=config)

    # callbacks
    total_steps = config['epochs'] * config['steps_per_epoch']

    if config['save_freq_epochs']:
        save_freq = config['save_freq_epochs'] * config['steps_per_epoch']
        mcc = ModelCheckpoint(mdl_path + '/epoch_{epoch}',
                              monitor='val_loss',
                              save_freq=save_freq)
    else:
        mcc = ModelCheckpoint(
            mdl_path, monitor='val_loss', save_best=True)

    tbc = TensorBoard(log_dir=log_dir, update_freq='epoch')
    callbacks = [tbc, mcc]

    loss, metrics, outputs = get_outputs_info(config)

    # mirrored strategy for multi gpu
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
    test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)

    # model
    with mirrored_strategy.scope():
        if args.model == 'rewcnn':
            model = build_rewcnn_from_config(config)
        elif args.model == 'rewbert':
            model = build_rewbert_from_config(config)
        elif args.model == 'rewmix':
            model = build_rewmix_from_config(config)
        else:
            raise ValueError('Unknown model %s' % args.model)
        model.summary()

        lr_decay = config['lr_decay']
        lr_schedule = None

        if lr_decay == 'linear':
            lr_schedule = CosineDecay(config['lr'],
                                      decay_steps=total_steps)

        elif lr_decay == 'cosine':
            lr_schedule = PolynomialDecay(config['lr'],
                                          decay_steps=total_steps,
                                          end_learning_rate=0)

        if lr_schedule:
            optimizer = tf.keras.optimizers.Adam(lr_schedule)

        else:
            optimizer = tf.keras.optimizers.Adam(config['lr'])

        model.compile(optimizer, loss=loss, metrics=metrics)

    # train
    _ = model.fit(train_ds,
                  validation_data=test_ds,
                  epochs=config["epochs"],
                  steps_per_epoch=config["steps_per_epoch"],
                  callbacks=callbacks,
                  validation_steps=config["validation_steps"])

    # extract and save tokenizer
    tok_path = "models/tokenizers/%s" % stub
    saved_model = tf.keras.models.load_model(
        mdl_path + '/epoch_%s' % config['epochs'])
    tokenizer = tf.keras.Model(
        saved_model.input, saved_model.get_layer('tokenizer').output)
    tokenizer.compile('adam', 'mse')
    tokenizer.summary()
    tokenizer.save(tok_path)


def main(args):
    tf_cap_memory()

    # config is a single json file
    if str(args.config).endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
        train(args, config)

    # config is a folder with multiple json file configs to run
    else:
        c_dir = os.listdir(args.config)
        config_paths = [args.config + f for f in c_dir if f.endswith('.json')]

        for config_path in config_paths:
            with open(config_path) as f:
                config = json.load(f)

            train(args, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training config')
    parser.add_argument('--config', '-c', help='config path')
    parser.add_argument('--model', '-m', help='model name e.g rewcnn')
    parser.add_argument('--output_dir', '-o', help='base output directory',
                        default='models')
    parser.add_argument('--dataset_bucket', '-p', help='gcs bucket of dataset',
                        default='retvec-internal')
    parser.add_argument('--dataset_path', '-d', help='gcs path to dataset',
                        default='retvec_datasets/word_model/dataset_v2/dataset')

    args = parser.parse_args()
    if not args.config or not args.dataset_path:
        parser.print_usage()
        quit()
    main(args)

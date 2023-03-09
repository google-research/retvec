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
from pathlib import Path
from time import time
from typing import Dict

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow_addons.optimizers import LAMB
from termcolor import cprint
from wandb.keras import WandbCallback

from retvec.tf.io import get_dataset_samplers, get_outputs_info
from retvec.tf.models.rewformer import build_rewformer_from_config
from retvec.tf.models.rewmlp import build_rewmlp_from_config
from retvec.tf.optimizers import WarmupCosineDecay
from retvec.tf.utils import tf_cap_memory


def train(args: argparse.Namespace, config: Dict) -> None:
    # save paths
    if args.experiment_name:
        model_name = args.experiment_name
    else:
        model_name = "%s-v%s" % (config["name"], config["version"])
    cprint("[Model: %s]" % (model_name), "yellow")
    cprint("|-epochs: %s" % config["train"]["epochs"], "blue")
    cprint("|-steps_per_epoch: %s" % config["train"]["steps_per_epoch"], "green")
    cprint("|-batch_size: %s" % config["train"]["batch_size"], "blue")
    stub = "%s_%s" % (model_name, int(time()))

    output_dir = Path(args.output_dir)
    mdl_path = output_dir / "mdl_ckpts" / stub
    log_dir = output_dir / "logs" / stub

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity="marinazh",
            group=config["model"]["type"],
            name=model_name,
        )
        wandb.config = config

    # dataset
    train_ds, test_ds = get_dataset_samplers(
        bucket=args.dataset_bucket,
        train_path=args.train_dataset_path,
        test_path=args.test_dataset_path,
        config=config,
    )

    # callbacks
    epochs = config["train"]["epochs"]
    steps_per_epoch = config["train"]["steps_per_epoch"]
    total_steps = epochs * steps_per_epoch
    save_freq_epochs = config["train"]["save_freq_epochs"]
    validation_steps = config["train"]["validation_steps"]

    if save_freq_epochs:
        save_freq = save_freq_epochs * steps_per_epoch
        mcc = ModelCheckpoint(mdl_path / "epoch_{epoch}", monitor="loss", save_freq=save_freq)
    else:
        mcc = ModelCheckpoint(mdl_path, monitor="loss", save_best=True)

    tbc = TensorBoard(log_dir=log_dir, update_freq="epoch")
    callbacks = [tbc, mcc]

    if args.wandb_project:
        callbacks.append(WandbCallback(save_model=False))

    loss, metrics, _ = get_outputs_info(config)

    # mirrored strategy for multi gpu
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # model
    with mirrored_strategy.scope():
        model_type = config["model"]["type"]
        if model_type == "rewformer":
            model = build_rewformer_from_config(config)
        elif model_type == "rewmlp":
            model = build_rewmlp_from_config(config)
        else:
            raise ValueError("Unknown model %s" % model_type)

        lr_schedule = WarmupCosineDecay(
            max_learning_rate=config["train"]["max_learning_rate"],
            total_steps=total_steps,
            warmup_steps=config["train"]["warmup_steps"],
            alpha=config["train"]["end_lr"] / config["train"]["max_learning_rate"],
        )

        if config["train"]["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(lr_schedule)

        if config["train"]["optimizer"] == "lamb":
            optimizer = LAMB(lr_schedule)

        model.summary()
        model.compile(optimizer, loss=loss, metrics=metrics)

    # train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_steps=validation_steps,
    )

    # extract and save tokenizer
    rewnet_path = Path(args.output_dir) / "rewnet" / stub
    embedding_path = Path(args.output_dir) / "embeddings" / stub
    results_path = Path(args.output_dir) / "results" / stub

    os.makedirs(rewnet_path, exist_ok=True)
    os.makedirs(embedding_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # check that model can be reloaded
    if save_freq_epochs:
        saved_model_path = mdl_path / f"epoch_{epochs}"
    else:
        saved_model_path = mdl_path

    if saved_model_path.exists():
        saved_model = tf.keras.models.load_model(saved_model_path)

        # save model without optimizer so it can be loaded with only tensorflow
        saved_model.save(rewnet_path, include_optimizer=False)

        # if tokenizer layer is already defined in the model, save just the embedding model
        if "tokenizer" in [l.name for l in model.layers]:
            embedding_model = tf.keras.Model(saved_model.layers[2].input, saved_model.get_layer("tokenizer").output)
            embedding_model.compile("adam", "mse")
            embedding_model.summary()
            embedding_model.save(embedding_path, include_optimizer=False)
            cprint(f"Saving RetVec embedding model to {embedding_path}", "blue")

    with open(results_path / "train_history.json", "w") as f:
        json.dump(history.history, f)

    with open(results_path / "train_config.json", "w") as f:
        json.dump(config, f)

    wandb.finish()


def main(args: argparse.Namespace) -> None:
    # grow gpu memory usage when neccessary
    tf_cap_memory()

    # config is a single json file or a folder
    if str(args.model_config).endswith(".json"):
        model_config_paths = [args.model_config]

    else:
        model_dir = Path(args.model_config)
        c_dir = sorted(os.listdir(model_dir))
        model_config_paths = [str(model_dir / f) for f in c_dir if f.endswith(".json")]

        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx else len(model_config_paths)
        model_config_paths = model_config_paths[start_idx:end_idx]

    for model_config_path in model_config_paths:
        with open(model_config_path) as f:
            model_config = json.load(f)
        with open(args.train_config) as f:
            train_config = json.load(f)

        config = model_config
        config["train"] = train_config
        train(args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetVec Training")
    parser.add_argument(
        "--train_config",
        "-c",
        help="train config path",
        default="./configs/train_full.json",
    )
    parser.add_argument(
        "--model_config", "-m", help="model config file or folder path", default="./configs/models/rewmlp-256.json"
    )
    parser.add_argument("--output_dir", "-o", help="base output directory", default="./experiments/")
    parser.add_argument(
        "--start_idx",
        "-s",
        type=int,
        help="start idx in alphabetically sorted experiment dir (zero-indexed, inclusive)",
        default=0,
    )
    parser.add_argument(
        "--end_idx", "-e", type=int, help="end idx in alphabetically sorted experiment dir (zero-indexed, exclusive)"
    )
    parser.add_argument(
        "--dataset_bucket",
        "-p",
        help="gcs bucket of dataset",
        default="retvec_datasets",
    )
    parser.add_argument(
        "--train_dataset_path",
        help="gcs path to training dataset",
        default="full/retvec_fasttext_deduplicated_v2/",
    )
    parser.add_argument(
        "--test_dataset_path",
        help="gcs path to testing dataset",
        default="full/retvec_fasttext_deduplicated_v2/",
    )
    parser.add_argument(
        "--wandb_project",
        "-w",
        default="REWNet-Pretraining",
        help="Wandb project to save to, none to disable.",
    )
    parser.add_argument(
        "--experiment_name",
        "-n",
        help="Experiment name. If none, uses default of [model_name]-[version]",
    )
    args = parser.parse_args()

    main(args)

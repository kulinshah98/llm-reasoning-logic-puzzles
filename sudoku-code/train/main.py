# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for Sudoku GPT experiments."""

import sys

from absl import app
from absl import flags
from absl import logging

from clu import platform

import jax
import tensorflow as tf
import wandb

import ml_collections
from ml_collections import config_flags

from train import train_and_evaluate

import pdb



sys.dont_write_bytecode = True

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Directory to store model data.')
_EXP_NAME = flags.DEFINE_string(
    'exp_name',
    None,
    'Experiment name.')
_CKPT_LOC = flags.DEFINE_string(
    'ckpt_loc',
    None,
    'Directory to restore model.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['workdir', 'exp_name'])


def get_config():
    """Get the default hyperparameter configuration.

    Returns:
    A ConfigDict object.
    """
  
    # Common configuration for all experiments.
    config = ml_collections.ConfigDict()

    # Dataset choice
    config.dataset = 'sudoku'

    # Sequence order
    config.seq_order = "solver-order"           ## Choices = ["fixed", "solver-order", "random"]

    # Training related parameters
    config.max_steps = 2**22
    config.dtype = jax.numpy.bfloat16
    config.minibatch_size = 64

    # Model related parameters
    config.block_size = 81
    config.seq_len = 3 * config.block_size
    config.vocab_size = 11

    # Model architecture
    config.num_heads = 8
    config.num_layers = 8
    config.emb_dim = 576
    config.qkv_dim = 576
    config.mlp_dim = 6 * config.emb_dim
    config.dropout_rate = 0.2
    config.attention_dropout_rate = 0.2

    # Training hyperparameters
    config.learning_rate = 0.0002  # Base learning rate.
    config.end_lr_factor = 0.2
    config.warmup_tokens = 10000
    config.weight_decay = 0.005
    config.resume_training = False

    # Other hyperparameters
    config.seed = 7
    config.save_checkpoint = True
    config.save_every_steps = 32000
    config.use_wandb = False
    config.wandb_project_name = 'sudoku'

    # Evaluation related parameters
    config.eval_every_steps = 32000
    config.eval_epochs = 5

    # Path to dataset
    config.train_puzzle_path = "datasets/train_sudoku_puzzles.npy"
    config.train_candidate_path = "datasets/train_sudoku_puzzles_candidate.npy"
    config.test_puzzle_path = "datasets/test_sudoku_puzzles.npy"
    config.test_candidate_path = "datasets/test_sudoku_puzzles_candidate.npy"

    return config


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    cfgs = get_config()
    if cfgs.resume_training: 
        assert _CKPT_LOC.value is not None
  
    if cfgs.use_wandb:
        wandb.init(project=cfgs.wandb_project_name, name=_EXP_NAME.value, config=cfgs)

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       _WORKDIR.value, 'workdir')
  
    logging.info(cfgs)

    cfgs.workdir = _WORKDIR.value
    cfgs.ckpt_loc = _CKPT_LOC.value
    train_and_evaluate.train_and_evaluate(cfgs, _WORKDIR.value)

    if cfgs.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    jax.config.config_with_absl()
    app.run(main)
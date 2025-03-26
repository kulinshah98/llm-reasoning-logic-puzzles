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

  Dataset choices:
    othello: For othello game
    sudoku: Sudoku game but fixed order (order: row-wise left to right)
    ordered-sudoku: Sudoku game data with the order of solver
    ordered-sudoku-wo-random-guessing-w-candidates-train-test: Uses sudoku games
              that can be solved with 7 human logics. It has train-test split.
              Does not contain examples with random guessing. Has penciling
              candidates for 10 locations and strategies used for each of the
              move.
  Returns:
    A ConfigDict object.
  """

  config = ml_collections.ConfigDict()
  config.max_n = 6
  config.max_m1 = 6

  ### Training related parameters
  config.max_steps = 2**21
  config.dtype = jax.numpy.bfloat16
  config.minibatch_size = 64
  
  config.max_seq_len = 600
  config.vocab_size = 20

  ### Model related parameters
  config.num_heads = 8
  config.num_layers = 8
  config.emb_dim = 576
  config.qkv_dim = 576
  config.mlp_dim = 6 * config.emb_dim
  config.dropout_rate = 0.2
  config.attention_dropout_rate = 0.2

  ### Training related parameters
  config.learning_rate = 0.0002  # Base learning rate.
  config.end_lr_factor = 0.2
  config.warmup_tokens = 5000
  config.weight_decay = 0.005
  config.optimizer = 'adamw'
  config.resume_training = False

  config.seed = 7
  config.save_checkpoint = True
  config.save_every_steps = 16000
  config.use_wandb = True

  ### Evaluation related parameters
  config.eval_every_steps = 16000
  config.eval_epochs = 5
  config.set_accuracy = "top-k"
  config.set_accuracy_top_k = 10
  config.beam_search_n = 1
  config.dataset_path = None

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
  train_and_evaluate.train_and_evaluate(cfgs, _WORKDIR.value, _EXP_NAME.value)

  if cfgs.use_wandb:
    wandb.finish()


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
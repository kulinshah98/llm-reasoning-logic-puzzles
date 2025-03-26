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


"""This file contains function that coordinates the training and evaluation of the model."""

import functools
import math

from absl import logging
from clu import metric_writers
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints

import jax
from jax import random
import jax.numpy as jnp

import numpy as np
import tensorflow as tf
import wandb

from train import data
from train import model
from train import trainer
from train import evaluater

import pdb

def log_hyperparams_tb(
    config, model_config, initial_variables, tf_summary_writer, expdir
):
  """Log hyperparameters to TensorBoard.

  Args:
    config: experiment's ConfigDict
    model_config: model's ConfigDict
    initial_variables: initial hyperparameter values
    tf_summary_writer: SummaryWriter object.

  Returns:
    The SummaryWriter object and the config.
  """
  config.num_model_parameters = sum(
      x.size for x in jax.tree_util.tree_leaves(initial_variables)
  )

  config_hyperparameters = [
      tf.convert_to_tensor([k, str(v)]) for k, v in config.items()
  ]
  model_config_hyperparameters = [
      tf.convert_to_tensor([k, str(v)])
      for k, v in model_config.__dict__.items()
  ]

  if config.use_wandb:
    wandb.init(project="llm-reasoning-logic-puzzles", name=expdir, config=config, dir="./logs/wandb")

  with tf_summary_writer.as_default():
    tf.summary.text(
        "Model hyperparameters", tf.stack(model_config_hyperparameters), step=0
    )
    tf.summary.text(
        "Config hyperparameters", tf.stack(config_hyperparameters), step=0
    )

  return tf_summary_writer, config



def train_and_evaluate(config, workdir, expdir):
  """
    Train and evaluate the model based on the provided configuration.

    Args:
        config: Configuration object containing hyperparameters and settings for training.
        workdir: Directory where the model checkpoints and logs will be saved.
        expdir: Directory for the current experiment's logs and outputs.
  """
  
  logging.info("Creating dataset iterators")
  
  config.max_seq_len = 457 ## Hardcoding max-length. TODO: fix this. 
  
  # Create iterators for training and evaluation datasets
  train_data_iter, train_ds_info = data.create_iter(config, config.minibatch_size, train=True)
  eval_data_iter, eval_ds_info = data.create_iter(config, config.minibatch_size, train=False)
  
  # Set configuration parameters based on dataset information
  config.vocab_size = train_ds_info["vocab_size"]
  config.vocab_dict = train_ds_info["vocab_dict"]
  config.reverse_vocab_dict = train_ds_info["reverse_vocab_dict"]
  
  logging.info("Finished creating dataset iterators")
  
  # Configure the model with the specified parameters
  model_config = model.TransformerConfig(
      dtype=config.dtype,
      vocab_size=config.vocab_size,
      max_seq_len=config.max_seq_len,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      emb_dim=config.emb_dim,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False
  )
  
  logging.info("train_config: %s", str(model_config.__dict__))
  
  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng, inference_rng = random.split(rng, num=3)

  rng, dropout_rng = jax.random.split(rng)
  input_shape = (config.minibatch_size, config.max_seq_len)

  # Initialize the model and get the initial variables
  net = model.TransformerLMHeadModel(model_config)
  rng_keys = {"params": init_rng, "dropout": dropout_rng}
  sample_out, initial_variables = jax.jit(
      net.init_with_output
      )(rng_keys, jnp.ones(input_shape, jnp.int32))
  
  # Get the training state and learning rate scheduler
  state, lr_scheduler_fn = trainer.get_state(config, net, initial_variables)
  if config.resume_training:
    state = checkpoints.restore_checkpoint(config.ckpt_loc, state)
    print("----------Restored model from", config.ckpt_loc, "-----------")
    
  writer = metric_writers.create_default_writer(
      workdir, asynchronous=False, just_logging=(jax.process_index() > 0))
  tf_summary_writer = tf.summary.create_file_writer(workdir)

  logging.info("config: %s", str(config.__dict__))
  state = jax_utils.replicate(state)

  dropout_rngs = jax.random.split(rng, jax.local_device_count())

  # Prepare the training step function for parallel execution
  p_train_step = jax.pmap(
      functools.partial(
          trainer.train_step,
          config=model_config,
          hyperparams=config,
          learning_rate_fn=lr_scheduler_fn),
      axis_name="batch",
      donate_argnums=(0,))

  # Prepare the forward pass function for parallel execution
  p_forward_pass = jax.pmap(
      functools.partial(evaluater.forward_pass,
                        config=model_config.replace(deterministic=True)),
      axis_name="batch", donate_argnums=(0,))
  
  hooks, report_progress, train_metrics = trainer.get_metrics_report_progress(
      config, workdir, writer)

  # Log hyperparameters to TensorBoard
  tf_summary_writer, config = log_hyperparams_tb(
      config, model_config, initial_variables, tf_summary_writer, expdir
  )

  with metric_writers.ensure_flushes(writer):
    for step in range(0, config.max_steps):
      if step % config.eval_every_steps == 0:
        print(step)
      
      # Perform a training step and update the state
      state, metrics = trainer.train_one_step(p_train_step, config, state,
                                              step, dropout_rngs, train_data_iter)
      
      for h in hooks:
        h(step)
            
      if step % config.eval_every_steps == 0:
        # Evaluate the model and print metrics
        eval_metrics = evaluater.get_eval_metrics(
            state, eval_data_iter, p_forward_pass, config, step)
        print(step, metrics["loss"], eval_metrics["acc"], flush=True)

        if config.use_wandb:
            log_dict = {'loss': metrics["loss"].mean(), 'learning rate': metrics["learning_rate"].mean()}
        
        for key in eval_metrics.keys():
          if config.use_wandb:
              log_dict[ "eval_" + key ] = np.array(eval_metrics[key]).mean()

        if config.use_wandb: wandb.log(log_dict, step=step)

        # Save the model checkpoint if required
        if config.save_checkpoint and step % config.save_every_steps == 0:
          checkpoints.save_checkpoint_multiprocess(
              workdir, jax_utils.unreplicate(state), step, keep=1, overwrite=True
          )
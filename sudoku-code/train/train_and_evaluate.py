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
from train import evaluater
from train import model
from train import trainer



def log_hyperparams_tb(
    config, model_config, initial_variables, tf_summary_writer
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
    # Calculate the total number of model parameters
    config.num_model_parameters = sum(
        x.size for x in jax.tree_util.tree_leaves(initial_variables)
    )

    # Convert hyperparameters to tensors
    config_hyperparameters = [
        tf.convert_to_tensor([k, str(v)]) for k, v in config.items()
    ]
    model_config_hyperparameters = [
        tf.convert_to_tensor([k, str(v)])
        for k, v in model_config.__dict__.items()
    ]

    # Log model hyperparameters to TensorBoard
    with tf_summary_writer.as_default():
        tf.summary.text(
            "Model hyperparameters", tf.stack(model_config_hyperparameters), step=0
        )
        tf.summary.text(
            "Config hyperparameters", tf.stack(config_hyperparameters), step=0
        )

    return tf_summary_writer, config



def train_and_evaluate(config, workdir):
    """The training and evaluation loops for the model.

    Args:
        config: experiment's config dictionary.
        workdir: directory to use for logging.
    """

    logging.info("Creating training and evaluator dataset iterator")
    train_data_iter = data.create_iter(config, config.minibatch_size, train=True)
    eval_data_iter = data.create_iter(config, config.minibatch_size, train=False)

    logging.info("Finished creating training dataset iterator")

    model_config = model.TransformerConfig(
        dtype=config.dtype,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        emb_dim=config.emb_dim,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        deterministic=False,
    )

    logging.info("train_config: %s", str(model_config.__dict__))
    print(str(model_config.__dict__), flush=True)

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, inference_rng = random.split(rng, num=3)

    # Initialize the model and get initial variables
    rng, dropout_rng = jax.random.split(rng)
    input_shape = (config.minibatch_size, config.seq_len)
    net = model.TransformerLMHeadModel(model_config)
    rng_keys = {"params": init_rng, "dropout": dropout_rng}
    sample_out, initial_variables = jax.jit(
        net.init_with_output
        )(rng_keys, jnp.ones(input_shape, jnp.int32))
    
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

    p_train_step = jax.pmap(
        functools.partial(
            trainer.train_step,
            config=model_config,
            hyperparams=config,
            learning_rate_fn=lr_scheduler_fn),
        axis_name="batch",
        donate_argnums=(0,))

    p_eval_step = jax.pmap(functools.partial(evaluater.eval_step,
                                             config=model_config.replace(deterministic=True)),
                                             axis_name="batch", donate_argnums=(0,))
    
    hooks, report_progress, train_metrics = trainer.get_metrics_report_progress(
      config, workdir, writer)

    tf_summary_writer, config = log_hyperparams_tb(
        config, model_config, initial_variables, tf_summary_writer
    )

    with metric_writers.ensure_flushes(writer):
        for step in range(0, config.max_steps):
            if step%10000 == 0:
                print("Step:", step, flush=True)

            state, metrics = trainer.train_one_step(p_train_step, config, state, 
                                                    step, dropout_rngs, train_data_iter)
            
            for h in hooks:
                h(step)

            if math.isnan(metrics["loss"][0]):
                print("The loss function became nan: This might be due to the choice of hyperparameters.")
                break

            if step % config.eval_every_steps == 0:
                eval_metrics = evaluater.get_eval_metrics(
                    state, eval_data_iter, p_eval_step, config)
                print(step, metrics["loss"], eval_metrics["acc"], flush=True)
                with tf_summary_writer.as_default():
                    tf.summary.scalar("loss", metrics["loss"].mean(), step=step)
                    tf.summary.scalar(
                        "learning rate", metrics["learning_rate"].mean(), step=step
                        )
                    
                    if config.use_wandb:
                        log_dict = {'loss': metrics["loss"].mean(), 'learning rate': metrics["learning_rate"].mean()}

                    for key in eval_metrics.keys():
                        tf.summary.scalar(
                            "eval_" + key, np.array(eval_metrics[key]).mean(), step=step
                        )
                        
                        if config.use_wandb:
                            log_dict[ "eval_" + key ] = np.array(eval_metrics[key]).mean()

                    if config.use_wandb: wandb.log(log_dict, step=step)


            if config.save_checkpoint and step % config.save_every_steps == 0:
                checkpoints.save_checkpoint_multiprocess(
                    workdir, jax_utils.unreplicate(state), step, keep=5, overwrite=True
                )

  
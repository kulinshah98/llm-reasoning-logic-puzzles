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

"""Transformer LM trainer."""

import functools

from clu import periodic_actions
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import numpy as jnp
import numpy as np
import optax
import ml_collections

from train import model


def get_state(config, net, initial_variables):
    """Get the train state given an experiment config, a model and initial variables.

    Args:
        config: A ConfigDict containing the configuration for the experiment.
        net: The model to use for training.
        initial_variables: The initial variables for the model.

    Returns:
        A tuple containing the train state and the learning rate schedule.
    """
    # Learning rate schedule
    lr_scheduler_fn = functools.partial(
        lr_scheduler,
        learning_rate=config.learning_rate,
        warmup_tokens=config.warmup_tokens,
        final_tokens=config.max_steps,
        config=config,
    )
    # Optimizer
    optim_fn = optax.adamw(
        lr_scheduler_fn, weight_decay=config.weight_decay, b1=0.9, b2=0.95
    )
    # Clip the gradients to prevent exploding gradients.
    optimizer = optax.chain(optax.clip_by_global_norm(1), optim_fn)

    # Initialize the train state
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=initial_variables["params"],
        tx=optimizer
        )

    return state, lr_scheduler_fn

def lr_scheduler(
    n_tokens: int, learning_rate: float, warmup_tokens: int, final_tokens: int,
    config: ml_collections.ConfigDict,
) -> float:
    """Learning rate scheduler, adapted from Mikhail Grankin.

    The learning rate schedule is cosine decay with a warmup period.

    The learning rate starts at 0 and linearly increases to the given learning
    rate over the warmup period. After the warmup period, the learning rate
    decays according to a cosine schedule, with the given learning rate as the
    maximum value.

    Args:
        n_tokens: The number of tokens processed so far.
        learning_rate: The initial learning rate.
        warmup_tokens: The number of tokens to warm up over.
        final_tokens: The total number of tokens to process.
        config: A ConfigDict containing the configuration for the learning rate
            schedule.

    Returns:
        The learning rate at the given point in the schedule.
    """
    # Decay the learning rate based on our progress.
    progress = (n_tokens - warmup_tokens) / max(
        1, final_tokens - warmup_tokens,
    )
    lr_mult = jnp.where(
        n_tokens < warmup_tokens,
        # Linear warmup.
        n_tokens / jnp.fmax(1, warmup_tokens),
        # Cosine learning rate decay.
        jnp.fmax(config.end_lr_factor, 0.5 * (1.0 + jnp.cos(np.pi * progress))),
    )
    return learning_rate * lr_mult


def get_metrics_report_progress(config, workdir, writer):
    """
    Get the metrics for reporting progress during training.

    Args:
        config: The configuration for the experiment.
        workdir: The directory for storing the logs.
        writer: The writer object for recording the metrics.

    Returns:
        hooks: List of hooks for tracking progress.
        report_progress: Object for reporting progress.
        train_metrics: List of training metrics.
    """
    hooks = []

    # Initialize the report progress object
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.max_steps, writer=writer)

    # Add metrics for profiling if the process index is 0
    if jax.process_index() == 0:
        hooks += [report_progress,
                  periodic_actions.Profile(logdir=workdir, num_profile_steps=5)]
    
    # Initialize the list of training metrics
    train_metrics = []
    
    return hooks, report_progress, train_metrics


def get_input_start_index(batch, config):
    inputs = jax.tree_util.tree_map(np.asarray, batch[0])
    puzzles = jax.tree_util.tree_map(np.asarray, batch[1])
    start_index = jax.tree_util.tree_map(np.asarray, batch[2])
    return inputs, puzzles, start_index


def train_one_step(p_train_step, config, state, step, dropout_rngs, train_data_iter):
    """
    Single step of the training loop.

    Args:
        p_train_step: The training step function.
        config: The experiment configuration.
        state: The train state.
        step: The step number.
        dropout_rngs: The dropout random number generator.
        train_data_iter: The iterator for the train data.

    Returns:
        The updated train state and train metrics.
    """
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
        # Get the next batch from the iterator
        batch = next(train_data_iter)
        # Extract the inputs and start index from the batch
        inputs, _, start_index = get_input_start_index(batch, config)
        # Shard the inputs and start index across the devices
        inputs = common_utils.shard(jax.tree_util.tree_map(np.asarray, inputs))
        start_index = common_utils.shard(jax.tree_util.tree_map(np.asarray, start_index))

        # Run the training step
        state, metrics, _ = p_train_step(
            state, inputs, start_index, dropout_rng=dropout_rngs
        )

    return state, metrics


def train_step(state, batch, start_index, config, 
               hyperparams, learning_rate_fn, dropout_rng=None):
    """One step of the training loop.

    Args:
        state: Train state.
        batch: Input batch.
        start_index: The starting index.
        config: Experiment config.
        hyperparams: Hyperparameter dictionary.
        learning_rate_fn: Learning rate function.
        dropout_rng: RNG used for dropout.

    Returns:
        A new train state, train metrics, and computed model predictions.
    """
    # Extract inputs and labels from the batch
    inputs = batch[:, :-1]
    label = batch[:, 1:]

    # Update dropout_rng
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    dropout_rng_dict = {"dropout": dropout_rng}

    def loss_fn(params):
        """Compute the loss function."""
        corrupted_inputs = inputs
        pred_logits = model.TransformerLMHeadModel(config).apply(
            {"params": params}, corrupted_inputs, rngs=dropout_rng_dict)

        label_one_hot = jax.nn.one_hot(label, num_classes=config.vocab_size)

        # The variables label_one_hot and pred_logits both are 3-dimensional tensors with
        # first axis corresponding to batch size, second correspondingn to sequence length 
        # and third corresponding to the row/column/value at a particular cell
        assert label_one_hot.shape == pred_logits.shape, ("one hot label shape",
                                                        label_one_hot.shape,
                                                        label.shape,
                                                        pred_logits.shape)
        
        # Calculate the cross-entropy loss along the last axis
        pred_logits_sol = pred_logits[:, :, :]
        label_one_hot_sol = label_one_hot[:, :, :]

        ce_loss = optax.softmax_cross_entropy(
            logits=pred_logits_sol[:, :, :], labels=label_one_hot_sol[:, :, :]
        )
        # assert ce_loss.ndim == 2, ("ce_loss", ce_loss.shape)

        # Apply masking to the loss
        mask = np.repeat(
            np.arange(len(ce_loss[0])).reshape(1, -1), len(ce_loss), axis=0
        )
        mask = (mask >= 3 * start_index)      
        
        avg_ce_loss = (ce_loss * mask).sum() / pred_logits.shape[0]
        
        return avg_ce_loss, pred_logits

    # Compute the learning rate and perform gradient descent
    step = state.step
    lr = learning_rate_fn(step)
    (loss, pred_logits), grads = jax.value_and_grad(loss_fn,
                                                  has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    
    # Update training metrics
    metrics = {
        "step": step, "loss": loss * inputs.shape[0], "learning_rate": lr,
        "pred_logits": pred_logits, "weights": inputs.shape[0]
    }
    
    return new_state, metrics, pred_logits

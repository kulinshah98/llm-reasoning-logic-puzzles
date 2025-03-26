import functools

from clu import periodic_actions
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import numpy as jnp
import numpy as np
import optax
import pdb

from train import model

def train_step(state, batch, start_index, config, hyperparams, learning_rate_fn,
               dropout_rng=None):
    """One step of the training loop.

    Args:
        state: train state.
        batch: input batch
        config: experiment config
        hyperparams: hyperparameter dictionary
        learning_rate_fn: learning rate function
        dropout_rng: rng to be used for dropout

    Returns:
        A new train state, train metrics and computed model predictions.
    """

    # Extract inputs and labels from the batch
    inputs = batch[:, :-1]
    label = batch[:, 1:]

    # Update dropout RNG with the current step
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    dropout_rng_dict = {"dropout": dropout_rng}

    def loss_fn(params):
        # Forward pass through the model
        pred_logits = model.TransformerLMHeadModel(config).apply(
            {"params": params}, inputs, rngs=dropout_rng_dict)

        # Convert labels to one-hot encoding
        label_one_hot = jax.nn.one_hot(label, num_classes=config.vocab_size)

        # Ensure the shapes of predictions and labels match
        assert label_one_hot.shape == pred_logits.shape, (
            "Mismatch in shapes", label_one_hot.shape, label.shape, pred_logits.shape)

        # Compute cross-entropy loss
        ce_loss = optax.softmax_cross_entropy(logits=pred_logits, labels=label_one_hot)

        # Create a mask to ignore losses before the start index
        mask = np.repeat(np.arange(len(ce_loss[0])).reshape(1, -1), len(ce_loss), axis=0)
        mask = (mask >= start_index)

        # Calculate the average cross-entropy loss
        avg_ce_loss = (ce_loss * mask).sum() / pred_logits.shape[0]

        return avg_ce_loss, pred_logits

    # Get the current step and learning rate
    step = state.step
    lr = learning_rate_fn(step)

    # Compute gradients and loss
    (loss, pred_logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Average gradients across all devices
    grads = jax.lax.pmean(grads, "batch")

    # Update the training state with the new gradients
    new_state = state.apply_gradients(grads=grads)
    metrics = {
        "step": step,
        "loss": loss * inputs.shape[0],
        "learning_rate": lr,
        "pred_logits": pred_logits,
        "weights": inputs.shape[0]
    }

    return new_state, metrics, pred_logits


def train_one_step(p_train_step, config, state, step, dropout_rngs, train_data_iter):
    """Single step of the training loop."""
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = next(train_data_iter)

        inputs = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch[0]))
        start_index = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch[2].reshape(-1, 1)))

        state, metrics, _ = p_train_step(state, inputs, start_index, dropout_rng=dropout_rngs)

    return state, metrics



def get_metrics_report_progress(config, workdir, writer):
  hooks = []

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.max_steps, writer=writer)

  if jax.process_index() == 0:
    hooks += [report_progress,
              periodic_actions.Profile(logdir=workdir, num_profile_steps=5)]
  train_metrics = []
  return hooks, report_progress, train_metrics



def lr_scheduler(n_tokens, learning_rate, warmup_tokens, final_tokens, config):
  """Learning rate scheduler, adapted from Mikhail Grankin.

  Args:
    n_tokens: Number of tokens processed so far.
    learning_rate: Initial learning rate.
    warmup_tokens: Number of tokens to warm up over.
    final_tokens: Total number of tokens to process.
    config: Configuration dictionary containing additional parameters.

  Returns:
    Adjusted learning rate based on the current progress.
  """

  # Decay the learning rate based on our progress.
  progress = (n_tokens - warmup_tokens) / max(
      1,
      final_tokens - warmup_tokens,
  )
  lr_mult = jnp.where(
      n_tokens < warmup_tokens,
      # Linear warmup.
      n_tokens / jnp.fmax(1, warmup_tokens),
      # Cosine learning rate decay.
      jnp.fmax(config.end_lr_factor, 0.5 * (1.0 + jnp.cos(np.pi * progress))),
  )
  return learning_rate * lr_mult



def get_state(config, net, initial_variables):
  """Get the train state given an experiment config, a model and initial variables.

  Args:
    config: experiment's configuration dictionary.
    net: the neural network model.
    initial_variables: initial variables for the model.

  Returns:
    state: the training state.
    lr_scheduler_fn: the learning rate scheduler function.
  """
  
  # Create a learning rate scheduler function
  lr_scheduler_fn = functools.partial(
      lr_scheduler,
      learning_rate=config.learning_rate,
      warmup_tokens=config.warmup_tokens,
      final_tokens=config.max_steps,
      config=config,
  )
  
  # Define the optimizer function based on AdamW
  optim_fn = optax.adamw(
    lr_scheduler_fn, weight_decay=config.weight_decay, b1=0.9, b2=0.95
  )
  optimizer = optax.chain(optax.clip_by_global_norm(1), optim_fn)

  # Create the training state
  state = train_state.TrainState.create(
      apply_fn=net.apply, params=initial_variables["params"],
      tx=optimizer
  )

  return state, lr_scheduler_fn
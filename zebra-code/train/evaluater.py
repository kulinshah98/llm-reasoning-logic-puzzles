from flax.training import common_utils
import jax
from jax import numpy as jnp
import numpy as np

from train import model

import pdb

def forward_pass(state, batch, config):
  pred_logits = model.TransformerLMHeadModel(config).apply(
      {"params": state.params}, batch)
  return pred_logits


def get_pred_logits(cur_input_seq, bs, state, p_forward_pass, config):
  # Create padding for the input sequence
  padding = np.zeros(
      (bs, config.max_seq_len - len(cur_input_seq[0])),
      dtype=np.int32,
  )
  
  # Concatenate the current input sequence with the padding
  concat_batch = np.hstack((cur_input_seq, padding))
  concat_batch = common_utils.shard(
      jax.tree_util.tree_map(np.asarray, concat_batch)
  )

  # Get predicted logits from the forward pass
  pred_logits = p_forward_pass(state, concat_batch)
  return pred_logits


def get_beam_search_candidates(
    input_seq, beam_search_candidates, start_index, min_start_index,
    state, p_forward_pass, pos, config, top_n_pos
):
  """
  Generate beam search candidates based on the input sequence and current state.
  """
  
  new_beam_candidate_list = []
  new_beam_candidate_likelihood_list = []
  
  for i in range(len(beam_search_candidates)):
    # Iterate through all the beam search candidates
    
    # predict the logits for row/column/value
    pred_logits = get_pred_logits(
        beam_search_candidates[i][0], input_seq.shape[0], 
        state, p_forward_pass, config
    )

    # Choose top beam_search_n most probable predictions
    max_pos = (
        pred_logits[:, :, pos, :]
        .argpartition(-top_n_pos, axis=-1)[
            :, :, -top_n_pos :
        ]
        .reshape((-1, top_n_pos))
    )
    log_likelihood = jax.nn.log_softmax(pred_logits[:, :, pos, :]).reshape(
        (-1, pred_logits.shape[-1])
    )
    log_likelihood = np.take_along_axis(log_likelihood, max_pos, 1)

    # Append all of the candidates in new_beam_candidate_list
    append_or_not = (start_index <= pos).squeeze()
    for j in range(top_n_pos):
      cur_candidate = beam_search_candidates[i]
      append_val = (append_or_not * max_pos[:, j] + 
                    (1 - append_or_not) * input_seq[:, pos + 1])
      new_beam_candidate = np.hstack(
          (cur_candidate[0], jnp.reshape(append_val, newshape=(-1, 1)))
      )

      append_log_likelihood = (append_or_not * log_likelihood[:, j] + 
                               (1 - append_or_not) * np.zeros_like(log_likelihood[:, j]))
      new_beam_candidate_likelihood = cur_candidate[1] + append_log_likelihood
      new_beam_candidate_likelihood_list.append(new_beam_candidate_likelihood)
      new_beam_candidate_list.append(
          (new_beam_candidate, new_beam_candidate_likelihood, cur_candidate[2])
      )

  # Likelihood list for new candidates
  new_beam_candidate_likelihood_list = np.stack(
      new_beam_candidate_likelihood_list, axis=0
  )
  assert new_beam_candidate_likelihood_list.shape == (
      len(beam_search_candidates) * top_n_pos,
      input_seq.shape[0],
  ), new_beam_candidate_likelihood_list.shape

  # Find index of top beam_search_n in new candidates
  new_beam_candidate_ind = new_beam_candidate_likelihood_list.argpartition(
      -top_n_pos, axis=0
  )[-top_n_pos :, :]
  assert new_beam_candidate_ind.shape == (
      top_n_pos,
      input_seq.shape[0],
  ), new_beam_candidate_ind.shape

  # Create the new list by truncating to top beam_search_n candidates
  truncated_candidate_list = []
  for i in range(top_n_pos):
    new_candidate = np.zeros_like(new_beam_candidate_list[0][0])
    new_candidate_likelihood = np.zeros_like(new_beam_candidate_list[0][1])
    new_candidate_success_pred = np.zeros_like(new_beam_candidate_list[0][2])

    for j in range(input_seq.shape[0]):
      index = new_beam_candidate_ind[i, j]
      new_candidate[j] = new_beam_candidate_list[index][0][j]
      new_candidate_likelihood[j] = new_beam_candidate_list[index][1][j]
      new_candidate_success_pred[j] = new_beam_candidate_list[index][2][j]

    truncated_candidate_list.append(
        (new_candidate, new_candidate_likelihood, new_candidate_success_pred)
    )

  return truncated_candidate_list

  


def get_accuracy(cur_input_seq, start_index, min_start_index, state, p_forward_pass, 
                input_seq, puzzle_sol, config, eval_metrics, step):
  # Initialize beam search with first candidate
  beam_search_candidates = [(cur_input_seq, np.zeros(len(cur_input_seq)), 
                           np.zeros(len(cur_input_seq)))]
  total = np.zeros(len(cur_input_seq))

  # Perform beam search for each position
  for i in range(min_start_index, config.max_seq_len - 1):
    beam_search_candidates = get_beam_search_candidates(input_seq, beam_search_candidates,
        start_index, min_start_index, state, p_forward_pass, i, config, config.beam_search_n)
    
    # Check accuracy of each candidate
    for candidate in beam_search_candidates:
      for j in range(len(candidate[0])):
        if input_seq[j][i] == int(config.vocab_dict["EXTRA"]):
          continue
        
        if start_index[j] < i and (i - start_index[j]) % 3 == 0:
          total[j] += 1
          try:
            row_num = config.reverse_vocab_dict[str(candidate[0][j][i-2])]
            col_num = config.reverse_vocab_dict[str(candidate[0][j][i-1])]
            val = config.reverse_vocab_dict[str(candidate[0][j][i])]
            assert puzzle_sol[j][int(row_num) + 1, int(col_num)] == int(val)
          except:
            pass
          else:
            candidate[2][j] += 1

  # Get sequence with maximum probability
  total = total / len(beam_search_candidates)
  max_prob_seq = np.zeros_like(beam_search_candidates[0][0])
  max_prob = np.zeros((len(beam_search_candidates), beam_search_candidates[0][1].shape[0]))
  
  for j, candidate in enumerate(beam_search_candidates):
    max_prob[j, :] = candidate[1]

  max_prob_seq_ind = max_prob.argmax(axis=0)
  sucess_pred = np.zeros(len(max_prob_seq))

  for i in range(len(max_prob_seq)):
    max_prob_seq[i] = beam_search_candidates[max_prob_seq_ind[i]][0][i]
    sucess_pred[i] = beam_search_candidates[max_prob_seq_ind[i]][2][i]
    
  eval_metrics["acc"].append(sucess_pred.sum() * 1.0 / total.sum())
  return eval_metrics, max_prob_seq



def check_valid_solution(predicted_seq, correct, input_seq, start_index_seq, config):
  rvkeys = config.reverse_vocab_dict.keys()
  
  model_answer = np.zeros_like(correct)
  # Iterate through the predicted sequence in steps of 3
  for i in range(int(start_index_seq)+1, config.max_seq_len, 3):

    # Debugging breakpoint for max sequence length
    if i == config.max_seq_len:
      pdb.set_trace()
    
    try:
      # Check if predicted values are valid keys in the reverse vocabulary
      if str(predicted_seq[i]) not in rvkeys or str(predicted_seq[i+1]) not in rvkeys or str(predicted_seq[i+2]) not in rvkeys:
        return 0
    except:
      pdb.set_trace()

    # Map predicted values to their corresponding row, column, and value
    row_num = config.reverse_vocab_dict[str(predicted_seq[i])]
    col_num = config.reverse_vocab_dict[str(predicted_seq[i+1])]
    val = config.reverse_vocab_dict[str(predicted_seq[i+2])]

    # Check for end tokens in the input sequence
    if input_seq[i] == 20:  # End tokens
      if np.sum(model_answer[1:, :] != correct[1:, :]) == 0:
        return 1

    # Validate that row, column, and value are numeric
    if not (row_num.isnumeric() and col_num.isnumeric() and val.isnumeric()):
      return 0
    
    # Ensure indices are within valid bounds
    if int(row_num) < 0 or int(col_num) < 0 or int(val) < 0:
      return 0
    
    if int(row_num) >= len(correct) - 1:
      return 0
    
    if int(col_num) >= len(correct[0]) or int(val) >= len(correct[0]):
      return 0
    
    # Check if the predicted value matches the correct value
    if correct[int(row_num) + 1, int(col_num)] != int(val):
      return 0
    
    # Update the model answer with the predicted value
    model_answer[int(row_num) + 1, int(col_num)] = int(val)
    
  try:
    # Check if the model answer matches the correct answer
    if np.sum(model_answer != correct) == 0:
      return 1
  except:
    pdb.set_trace()
  
  return 0



def get_eval_metrics(state, eval_data_iter, p_forward_pass, config, step):
  # Initialize a dictionary to store evaluation metrics
  eval_metrics = {
      "acc": [],
      "acc_complete_puzzle": [],
  }

  # Loop through the evaluation epochs
  for eval_epoch in range(config.eval_epochs):
    with jax.profiler.StepTraceAnnotation("eval", step_num=eval_epoch):
      
      # Get the next batch of evaluation data
      batch_tuple = next(eval_data_iter)
      
      input_seq = np.array(batch_tuple[0])  # Input sequences
      puzzle_sol = np.array(batch_tuple[1])  # Corresponding puzzle solutions
      start_index = np.array(batch_tuple[2]).reshape(-1, 1)  # Start indices reshaped
      
      min_start_index = int(np.min(start_index))  # Minimum start index
      cur_input_seq = input_seq[:, :(min_start_index + 1)]  # Current input sequence based on min start index

      # Calculate accuracy and get the sequence with maximum probability
      eval_metrics, max_prob_seq = get_accuracy(
          cur_input_seq, start_index, min_start_index, state,
          p_forward_pass, input_seq, puzzle_sol, config, eval_metrics, step
      )

      complete_correct_puzzle = 0
      
      # Check each predicted sequence for validity
      for j in range(len(max_prob_seq)):
        correct_or_not = check_valid_solution(max_prob_seq[j], puzzle_sol[j], input_seq[j],
                                                        start_index[j], config)
        complete_correct_puzzle += correct_or_not  # Count complete correct puzzles

      # Append the accuracy of complete puzzles to the metrics
      eval_metrics["acc_complete_puzzle"].append(
        complete_correct_puzzle * 1.0 / len(cur_input_seq)
      )
      
      # Print metrics every 5 epochs
      if eval_epoch % 5 == 4:
        print(np.array(eval_metrics["acc_complete_puzzle"]).mean())
        
  
  return eval_metrics  # Return the evaluation metrics
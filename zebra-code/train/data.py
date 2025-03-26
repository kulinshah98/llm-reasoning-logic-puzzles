import itertools
import pickle
from nltk.lm import Vocabulary

import jax
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import gfile


def create_dataset(config, bs, train):
  """
  Creates a TensorFlow dataset and gathers dataset information.

  Args:
    config: Configuration object containing dataset parameters.
    bs: Batch size for the dataset.
    train: Boolean flag indicating whether the dataset is for training or testing.

  Returns:
    A tuple containing:
      - tf_ds: A TensorFlow dataset object.
      - ds_info: A dictionary with dataset information including:
        - max_seq_len: Maximum sequence length in the dataset.
        - vocab_size: Size of the vocabulary.
        - vocab_dict: Dictionary mapping vocabulary tokens to their respective indices.
        - reverse_vocab_dict: Dictionary mapping indices back to their respective vocabulary tokens.
  """
  ds, output_types, output_shapes = None, None, None
  
  # Create an instance of ZebraDataset with the given configuration and training flag
  ds = ZebraDataset(config, train=train)
  
  # Define the data types for the output of the dataset generator
  output_types = (tf.int16, tf.int16, tf.int32, 
                  tf.int32, tf.int32, tf.int32)
  
  # Define the shapes for the output of the dataset generator
  output_shapes = (tf.TensorShape([config.max_seq_len]),      
                   tf.TensorShape([config.max_m1 + 1, config.max_n]), 
                   tf.TensorShape([]), tf.TensorShape([]),
                   tf.TensorShape([]), tf.TensorShape([]))
  
  # Create a TensorFlow dataset from the generator with specified output types and shapes
  tf_ds = tf.data.Dataset.from_generator(
      generator=ds, output_types=output_types, output_shapes=output_shapes)
  
  tf_ds = tf_ds.repeat()
  tf_ds = tf_ds.shuffle(8 * config.minibatch_size, seed=0)
  tf_ds = tf_ds.batch(bs)
  
  # Collect dataset information such as maximum sequence length, vocabulary size, and dictionaries
  ds_info = {"max_seq_len": ds.max_seq_len, 
             "vocab_size": ds.vocab_size + 2, 
             "vocab_dict": ds.vocab_dict, 
             "reverse_vocab_dict": ds.reverse_vocab_dict}
  
  # Return the TensorFlow dataset and the dataset information
  return tf_ds, ds_info


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  def _prepare(x):
    return x._numpy()
  
  return jax.tree_map(_prepare, xs)



def create_iter(config, bs, train):
  """
  Args:
    config: Configuration object containing dataset parameters.
    bs: Batch size for the dataset.
    train: Boolean flag indicating whether to use the training dataset.

  Returns:
    it: An iterator that yields batches of numpy arrays.
    ds_info: A dictionary with dataset information including:
      - max_seq_len: Maximum sequence length in the dataset.
      - vocab_size: Size of the vocabulary.
      - vocab_dict: Dictionary mapping vocabulary tokens to their respective indices.
      - reverse_vocab_dict: Dictionary mapping indices back to their respective vocabulary tokens.
  """
  # Create a TensorFlow dataset and dataset information using the provided configuration, batch size, and training flag
  tf_ds, ds_info = create_dataset(config, bs, train=train)
  
  # Convert the TensorFlow dataset to an iterator of numpy arrays
  it = map(prepare_tf_data, tf_ds)
  
  return it, ds_info


class ZebraDataset():
  def __init__(self, config, train):
    self.config = config
    
    # Load the dataset from a pickle file based on the train flag
    self.data = self.load_pickle_file(train)

    # Initialize a set with the vocabulary from the first data entry
    vocab_set = set(self.data[1][0])
    
    # Initialize the maximum sequence length
    self.max_seq_len = 0
    
    # Iterate over the dataset to update the vocabulary set and determine the maximum sequence length
    for i in range(len(self.data)):
      vocab_set.update(self.data[i][0])  # Update the vocabulary set with characters from the current sequence
      self.max_seq_len = max(self.max_seq_len, len(self.data[i][0]))  # Update the maximum sequence length if needed

    # Create a sorted list of unique vocabulary tokens
    self.vocab = sorted(list(set(vocab_set)))
    
    # Determine the size of the vocabulary
    self.vocab_size = len(self.vocab)

    # Create a dictionary mapping each vocabulary token to a unique index starting from 1
    self.vocab_dict = {self.vocab[i]: i + 1 for i in range(0, len(self.vocab))}
    
    # Create a reverse dictionary mapping indices back to their respective vocabulary tokens
    self.reverse_vocab_dict = {str(i + 1): self.vocab[i] for i in range(0, len(self.vocab))}

    # Add an "EXTRA" token to the vocabulary and its reverse mapping
    self.vocab_dict["EXTRA"] = (self.vocab_size + 1)
    self.reverse_vocab_dict[str(self.vocab_size + 1)] = "EXTRA"

    # Store the token ID for the "ANSWER" token
    self.answer_token_id = self.vocab_dict["ANSWER"]
  
  
  def load_pickle_file(self, train):
    if train:
      file_name = "train.pkl"
    else:
      file_name = "test.pkl"
    
    with open(self.config.dataset_path + file_name, "rb") as f:
      arr = pickle.load(f)

    return arr
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    # Convert the sequence of characters at the given index to their corresponding vocabulary indices
    seq = [self.vocab_dict[char] for char in self.data[idx][0]] 
    # Ensure the sequence length does not exceed the maximum sequence length
    assert len(seq) <= self.max_seq_len
    # Append "EXTRA" tokens to the sequence to match the maximum sequence length
    appended_seq = np.hstack((np.array(seq), self.vocab_dict["EXTRA"] * np.ones(self.config.max_seq_len - len(seq))))
    
    # Initialize an answer table with zeros, with dimensions based on config parameters
    appended_answer_table = np.zeros((self.config.max_m1 + 1, self.config.max_n), dtype=np.int16)
    # Get the current dimensions of the answer table from the data
    cur_m1, cur_n = np.array(self.data[idx][1]).shape 
    # Fill the initialized answer table with the actual data
    appended_answer_table[:cur_m1, :cur_n] = self.data[idx][1]
    
    # Find the index of the "ANSWER" token in the appended sequence
    answer_id = np.where(appended_seq == self.answer_token_id)
    # Ensure there is exactly one "ANSWER" token in the sequence
    assert len(answer_id[0]) == 1

    # Return the processed sequence, answer table, index of "ANSWER" token, original sequence length, and dimensions of the answer table
    return appended_seq, appended_answer_table, answer_id[0][0], len(seq), cur_n, cur_m1 - 1
  
  
  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i)


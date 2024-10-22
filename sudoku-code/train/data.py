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

"""Data loading procedure for Othello and Sudoku game.
"""

import itertools
import pickle

import jax
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import gfile

import pdb



def create_dataset(config, bs, train):
    """Create Sudoku dataset according to the config.

    Args:
        config: a config object containing the hyparameters for the dataset
            creation.
        bs: batch size
        train: whether the dataset is for train or eval

    Returns:
        a tf.data.Dataset object
    """
    ds, output_types, output_shapes = None, None, None
    ds = SudokuDataset(config, train=train)
    # The output types of the dataset are (tf.int32, tf.int32, tf.int32)
    # which represent the sequence of moves, the solution of the puzzle and
    # the start index of the sequence.
    output_types = (tf.int32, tf.int32, tf.int32)
    # The output shapes of the dataset are (seq_len, block_size, 1)
    # where seq_len is the length of the sequence, block_size is the size of
    # the block (i.e. the number of cells in the puzzle) and 1 is the number of
    # output features.
    output_shapes = (
        tf.TensorShape([config.seq_len]),
        tf.TensorShape([config.block_size]),
        tf.TensorShape([1]),
    )

    # Create a tf.data.Dataset object from the generator.
    tf_ds = tf.data.Dataset.from_generator(
        generator=ds, output_types=output_types, output_shapes=output_shapes)

    # Repeat the dataset indefinitely.
    tf_ds = tf_ds.repeat()
    # Shuffle the dataset with a buffer size of 8 * bs and a seed of 0.
    tf_ds = tf_ds.shuffle(8 * config.minibatch_size, seed=0)
    # Batch the dataset with a batch size of bs.
    tf_ds = tf_ds.batch(bs)
    return tf_ds



def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    def _prepare(x):
        return x._numpy()  # pylint: disable=protected-access
    
    return jax.tree_map(_prepare, xs)


def create_iter(config, bs, train):
    tf_ds = create_dataset(config, bs, train=train)
    it = map(prepare_tf_data, tf_ds)
    return it

class SudokuDataset:
    """Sudoku dataset."""
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.preprocess_sudoku()


    def convert_to_fixed_or_random_order(self, inputs, start_index):
        """Convert the sequence of moves to either a fixed or random order.

        Args:
            inputs: a numpy array of shape (num_puzzles, seq_len) containing the
                sequence of moves for each puzzle
            start_index: a numpy array of shape (num_puzzles, 1) containing the starting
                index for each puzzle

        Returns:
            transformed_input: a numpy array of shape (num_puzzles, seq_len) containing the
                sequence of moves for each puzzle in either a fixed or random order
        """
        transformed_input = np.zeros_like(inputs)
        
        for i in range(len(inputs)):
            cur_seq = inputs[i]
            cur_start_index = start_index[i, 0]
            
            # Split the sequence into input and output prompts
            inp_prompt = cur_seq[ :(3 * cur_start_index) ].reshape(-1, 3)
            out_prompt = cur_seq[ (3 * cur_start_index): ].reshape(-1, 3)
            
            # Sort the input prompts in a fixed order
            if self.config.seq_order == "fixed":
                transformed_input[i, :(3 * cur_start_index) ] = inp_prompt[ np.lexsort( inp_prompt[:, ::-1].T ) ].flatten()
            # Randomly shuffle the input prompts
            elif self.config.seq_order == "random":
                transformed_input[i, :(3 * cur_start_index) ] = np.random.permutation(inp_prompt).flatten()
            
            # Sort the output prompts in a fixed order
            if self.config.seq_order == "fixed":
                transformed_input[i, (3 * cur_start_index): ] = out_prompt[ np.lexsort( out_prompt[:, ::-1].T ) ].flatten()
            # Randomly shuffle the output prompts
            elif self.config.seq_order == "random":
                transformed_input[i, (3 * cur_start_index): ] = np.random.permutation(out_prompt).flatten()
        
        return transformed_input

    def get_puzzles_start_index(self, path):
        """Get the puzzles, start index, and inputs from a given path.

        Args:
            path: the path to the file containing the puzzles

        Returns:
            inputs: a numpy array of shape (num_puzzles, seq_len) containing the
                sequence of moves for each puzzle
            puzzles: a numpy array of shape (num_puzzles, block_size) containing
                the solution of each puzzle
            start_index: a numpy array of shape (num_puzzles, 1) containing the
                start index of the sequence for each puzzle
        """
        with gfile.Open(path, "rb") as f:
            inputs_with_start_index = np.load(f)
        start_index = inputs_with_start_index[:, 0]  # Get the start index

        inputs = inputs_with_start_index[:, 1:]  
        # Delete the column corresponding to the set of strategies
        inputs = np.delete( inputs, np.arange(81) * 4 + 3, axis=1)
        
        puzzles = np.zeros((len(inputs), 81), dtype=np.int8)  # Initialize puzzles
        for j in range(81):
            cell_id = inputs[:, 3 * j] * 9 + inputs[:, 3 * j + 1]  # Get the cell id
            puzzles[np.arange(len(inputs)), cell_id] = inputs[:, 3 * j + 2]  # Set the puzzle
        
        return inputs, puzzles, start_index.reshape(-1, 1)
    
    
    def preprocess_sudoku(self):
        """Preprocess the sudoku for train and test datasets.
        
        Depending on the `train` flag, this method loads and processes the
        sudoku puzzles and their start indices from the appropriate paths, and
        optionally converts them to a fixed or random order based on the 
        configuration.
        """
        if self.train is True:
            # Load train puzzles, inputs, and start indices
            self.train_inputs, self.train_puzzles, self.train_start_index = (
                self.get_puzzles_start_index(self.config.train_puzzle_path)
            )
            # Convert train inputs to fixed or random order if specified
            if self.config.seq_order in {"fixed", "random"}:
                self.train_inputs = self.convert_to_fixed_or_random_order(self.train_inputs, self.train_start_index)
        
        elif self.train is False:
            # Load evaluation puzzles, inputs, and start indices
            self.eval_inputs, self.eval_puzzles, self.eval_start_index = (
                self.get_puzzles_start_index(self.config.test_puzzle_path)
            )
            # Convert evaluation inputs to fixed or random order if specified
            if self.config.seq_order in {"fixed", "random"}: 
                self.eval_inputs = self.convert_to_fixed_or_random_order(self.eval_inputs, self.eval_start_index)

    def __len__(self):
        if self.train is True:
            return len(self.train_puzzles)
        elif self.train is False:
            return len(self.eval_puzzles)
        
    def __getitem__(self, idx):
        """Returns the training or evaluation data at the given index.

        Each train_input and eval_input sequence is 243 size long. 
        For each cell of a Sudoku puzzle, there is (row, column, value) in each train_input sequence. 
        train_puzzles/eval_puzzles is 81 size long and contains solution of the corresponding puzzle.
        - start_index denotes the number of non-empty cells in the puzzle (and is 
            used to determine from when the model starts predicting the solution).

        Args:
            idx: The index of the data to retrieve.

        Returns:
            A tuple containing the input sequence, puzzle, and start index.
        """
        if self.train is True:
            return (
                self.train_inputs[idx, :],  # Input sequence
                self.train_puzzles[idx, :],  # Puzzle
                self.train_start_index[idx],  # Denotes number of non-empty cells in the puzzle
            )
        elif self.train is False:
            return (
                self.eval_inputs[idx, :],  # Input sequence
                self.eval_puzzles[idx, :],  # Puzzle
                self.eval_start_index[idx],  # Denotes number of non-empty cells in the puzzle
            )
            
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

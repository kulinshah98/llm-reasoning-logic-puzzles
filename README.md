
# Causal language modeling can elicit search and reasoning capabilities on logic puzzles
## Official implementation of the paper


---- 

## Installing the environment

The recommended way to run the code is with an Anaconda/Miniconda environment. First, clone the repository:

```
git clone https://github.com/kulinshah98/llm-reasoning-logic-puzzles.git
```

Then, create a new Anaconda environment and install the dependencies:
```
conda env create -f environment.yml -n logic_puzzles
```
---- 
## Preparing datasets

The dataset for the Sudoku and Zebra both puzzles is given in [Link](https://drive.google.com/drive/folders/1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW?usp=sharing). 

### Sudoku dataset

- `Sudoku-train-data.npy` contains around 1.8M suodku puzzles and `Sudoku-test-data.npy` contains 0.1M sudoku puzzles. Both the dataset only contains puzzles with the unique solutions. 
- Each example of the dataset contains 325 entries: 
    - First value: Number of filled cells in that example puzzle
    - Rest of 324 values (324 values = 4 values corresponding to each of the cell in $9 \times 9$ puzzle) can be divided in two parts. Each example first contains the information about the cells given in the puzzle and followed by the information about the empty cells of the puzzle. 
    - The information (4 values) about each of the cell in the puzzle is given in the form of (row of the cell, column of the cell, correct value of the cell, strategy/list of strategy that needs to be applied to get the correct value). 
    - Strategy id: (0) the cell is given, (2) the cell is filled with Lone single, (3) Hidden single, (4) Naked pair, (5) Naked Triplet, (6) Locked Candidate, (7) XY Wing,
    (8) Unique Rectangle


### Zebra dataset

- `zebra-train-data.pkl` contains around 1.5M puzzles and `zebra-test-data.pkl` contains around 0.1M puzzles with number of entities (houses) and number of attributes (properties) in the range of 3 to 6. 
- Each example contains a list with 3 elements. 
    - First element of the list contains the list of the clues/hints for the puzzle in the symbolic language. 
    - Second element of the list contains the solution box. 
    - Third element of the list contains the order in which solver decides values in the solution box. Intuitively, the solver tries to fill the easy-to-decode cells. 





---- 
## Training a model on the sudoku puzzle

To train a new model, set the experiment name in {EXP_NAME} and work directory name in {WORKDIR} in scripts/train.sh file and run the following two commands:

```
cd sudoku-code
bash scripts/train.sh
```

To change the hyperparameters of the training, check `sudoku-code/train/main.py` file.


### Acknowledgement

This code is inspired from the following repositories.

1. JAX/Flax implementation of minGPT ([Link](https://github.com/brentyi/minGPT-flax/tree/master))
2. Dokusan Sudoku solver ([Link](https://github.com/unmade/dokusan))
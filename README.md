
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
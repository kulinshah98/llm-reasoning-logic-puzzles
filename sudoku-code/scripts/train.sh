EXP_NAME=sudoku_training
WORKDIR=./logs/$EXP_NAME

# Removes pycache files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# Runs the train/main.py file with appropriate arguments
python -m train.main --workdir=$WORKDIR --exp_name=$EXP_NAME 
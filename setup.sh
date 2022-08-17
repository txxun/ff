#!/bin/bash

set -e

unset PYTHONPATH

CONDA_HOME=$HOME/miniconda3/

CONDA_EXEC=$CONDA_HOME/bin/conda


if ! [ -f $CONDA_EXEC ]; then
    echo "----------------------"
    echo "Downloading miniconda!"
    echo "----------------------"

    echo $CONDA_EXEC

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh -b

    rm -rf Miniconda3-latest-Linux-x86_64.sh
    
fi

eval "$($CONDA_EXEC shell.bash hook)"

conda config --set always_yes yes

if [ "$venv" = "" ]; then
    venv=".venv_ff"
fi

VENV_PATH=$CONDA_HOME/envs/$venv

if ! [ -d $VENV_PATH ]; then
    echo "----------------------------"
    echo "Creating virtual environment"
    conda create --name $venv python=3.8
fi

conda activate $venv

echo "-----------------------"
echo "Installing requirements"
echo "-----------------------"


pip install --upgrade cython

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

#pip install -r requirements.txt

#pip install -e .

echo "---------"
echo "All done!"
echo "---------"



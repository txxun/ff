#!/bin/bash

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"


if [ "$venv" = "" ]; then
    venv=".venv_ff"
fi

export VENV_PATH=$HOME/miniconda3/envs/$venv

conda activate $venv

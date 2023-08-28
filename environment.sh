#!/bin/bash

# change as you wish
default_name="vision"
conda_packages=(
    "pip ipykernel ipywidgets autopep8 black"
    "numpy scipy matplotlib"
    "pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia"
)
pip_packages=()

# prompt
echo -e "A new conda environment will be created: $default_name\n"
echo -e "  - Press ENTER to confirm the environment name"
echo -e "  - Press CTRL-C to abort the environment setup"
echo -e "  - Or specify a different environment name\n"

# read environment name
read -p "[$default_name] >>> " name
if [ -z "$name" ]; then
    name="$default_name"
fi

# initialize conda
__conda_setup="$($HOME/anaconda3/bin/conda shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# create conda environment
conda create -n "$name" -y

# check if successful
exit_status=$?
if [ $exit_status -ne 0 ]; then
    exit $exit_status
fi

# activate conda environment
conda activate "$name"

# install conda packages
for packages in "${conda_packages[@]}"; do
    conda install $packages -y
done

# install pip packages
for packages in "${pip_packages[@]}"; do
    pip install $packages
done

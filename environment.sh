#!/bin/bash

conda install pip autopep8 black ipykernel ipywidgets numpy scipy matplotlib -y
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -y

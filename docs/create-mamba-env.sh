#!/bin/bash

set -xe

# install mamba
yum install -y wget bzip2
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

# install python deps
micromamba create -y -f ./environment.yml
micromamba activate cmip6-downscaling-docs

#!/bin/bash

set -x

# install mamba
yum install -y wget bzip2
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

# install python deps
micromamba create -y -f ./environment.yml
micromamba activate docs
pip install ..

# build json content
make json
ls _build/json/*

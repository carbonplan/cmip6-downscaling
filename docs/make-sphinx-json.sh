#!/bin/bash

set -xe

source ~/.bashrc

micromamba activate cmip6-downscaling-docs

# build json content
make json
ls _build/json/*

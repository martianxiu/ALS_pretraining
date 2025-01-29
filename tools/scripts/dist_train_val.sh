#!/usr/bin/env bash

set -x # print all executed command 
NGPUS=$1 # first argument
PY_ARGS=${@:2} # Python file arugments. @: all. :2: start from the second one

# DDP (singe node)
torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} train_val.py --launcher pytorch ${PY_ARGS}
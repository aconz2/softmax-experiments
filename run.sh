#!/usr/bin/env bash

set -e

clang -g -Wall -march=native -O2 -ffast-math softmax.c -lm -lsleef -o softmax
clang -g -Wall -march=native -O2 -ffast-math presum.c -lm -lsleef -o presum

taskset -c 0 ./presum
taskset -c 0 ./softmax

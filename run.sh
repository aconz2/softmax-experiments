#!/usr/bin/env bash

set -e

clang -DNDEBUG -g -Wall -march=native -O2 -ffast-math softmax.c -lm -lsleef -o softmax
clang -DNDEBUG -g -Wall -march=native -O2 -ffast-math presum.c -o presum
clang -DNDEBUG -g -Wall -march=native -O2 search.c -o search

taskset -c 30 ./presum
taskset -c 30 ./softmax
taskset -c 30 ./search

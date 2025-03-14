#!/usr/bin/env bash

set -e

clang -DNDEBUG -g -Wall -march=native -O2 -ffast-math softmax.c -lm -lsleef -o softmax
clang -DNDEBUG -g -Wall -march=native -O2 -ffast-math presum.c -lm -lsleef -o presum
clang -DNDEBUG -g -Wall -march=native -O2 search.c -o search

taskset -c 0 ./presum
taskset -c 0 ./softmax
taskset -c 0 ./search

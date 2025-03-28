#!/usr/bin/env bash

set -e

CC=${CC:-clang}

$CC -DNDEBUG -g -Wall -march=native -funsafe-math-optimizations -O2 softmax.c -lm -lsleef -o softmax
$CC -DNDEBUG -g -Wall -march=native -funsafe-math-optimizations -O2 presum.c -o presum
$CC -DNDEBUG -g -Wall -march=native -O2 search.c -o search

taskset -c 30 ./presum
taskset -c 30 ./softmax
taskset -c 30 ./search

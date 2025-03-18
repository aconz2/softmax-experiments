#!/usr/bin/env bash

set -e

clang -DRUNTEST -fno-sanitize-recover=all -fsanitize=undefined,address,leak -g -Wall -march=native -O2 -ffast-math softmax.c -lm -lsleef -o softmax-test
clang -DRUNTEST -fno-sanitize-recover=all -fsanitize=undefined,address,leak -g -Wall -march=native -O2 -ffast-math presum.c -lm -o presum-test
clang -DRUNTEST -fno-sanitize-recover=all -fsanitize=undefined,address,leak -g -Wall -march=native -O2 search.c -o search-test

./presum-test
./softmax-test
./search-test

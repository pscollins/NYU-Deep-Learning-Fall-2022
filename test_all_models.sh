#!/bin/bash

set -x
set -e

source env/bin/activate

for checkpoint in {0409999,0404999,0414999,0419999,0399999}; do
    pushd .
    cd /home/patrick/code/NYU-Deep-Learning-Fall-2022/data/weights
    git switch checkin-${checkpoint}
    popd

    for size in {400,410,420,430,440,450}; do
        python3 ./eval_with_overrides.py  INPUT.MIN_SIZE_TEST $size INPUT.MAX_SIZE_TEST 0 &> /tmp/result_checkin_${checkpoint}_$size
    done
done

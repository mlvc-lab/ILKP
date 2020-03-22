#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2
    do
        # python3 main.py $data -a $model -j 8 -C -g 0 -E --ckpt ckpt_best.pth
        for ver in v2
        do
            python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v $ver --analysis
            # python3 main.py $data -a $model -j 8 -C -g 0 -E --ckpt ckpt_best_$ver.pth -N
        done
    done
done

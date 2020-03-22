#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for layer in 20 32 44 56 110
    do
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt ckpt_best_v2a.pth -N -v v2a --lr 0.01
        rm -f checkpoint/*/*/ckpt_rt*_v*_epoch_*.pth
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 -E -N --ckpt ckpt_rt1_v2a_best.pth
    done
done

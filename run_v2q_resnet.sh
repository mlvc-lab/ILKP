#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for layer in 20 32 44 56 110
    do
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -N -v v2q --qb 8 
        rm -f checkpoint/*/*/ckpt_new_v*_q*_epoch_*.pth
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 -E -N --ckpt ckpt_new_v2q_q8_best.pth --qb 8
    done
done

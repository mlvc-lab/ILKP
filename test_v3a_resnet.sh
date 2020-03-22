#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for layer in 20 32 44 56 110
    do
        for bsize in 2 4 8 16
        do
            python3 find_similar_kernel.py $data -a resnet --layer $layer --ckpt ckpt_best.pth -v v3a -d $bsize
            python3 main.py $data -a resnet --layers $layer -j 8 -C -g 0 -E -N --ckpt "ckpt_best_v3a_d"$bsize".pth" -d $bsize
        done
    done
done

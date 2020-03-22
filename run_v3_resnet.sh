#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for layer in 20 32 44 56 110
    do
        for bsize in 2 4 8 16
        do
            python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -N -v v3 -d $bsize
            rm -f checkpoint/*/*/ckpt_new_v*_d*_epoch_*.pth
            python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 -E -N --ckpt "ckpt_new_v3_d"$bsize"_best.pth" -d $bsize
        done
    done
done

#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for layer in 20 32 44 56 110
    do
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt ckpt_best_v1.pth -N -v v1 --lr 0.01
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt ckpt_best_v2.pth -N -v v2 --lr 0.01
        rm -f checkpoint/*/*/ckpt_rt*_v*_epoch_*.pth
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 -E --ckpt ckpt_rt1_v1_best.pth -N
        python3 main.py $data -a resnet --layer $layer -j 8 -C -g 0 -E --ckpt ckpt_rt1_v2_best.pth -N
    done
done

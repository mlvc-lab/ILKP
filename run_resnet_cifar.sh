#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in resnet
    do
        for layers in 20 32 44 56 110
        do
            python3 main.py $data -a $model --layers $layers -j 8 -C -g 0 1 2 3 -b 256
            rm -f checkpoint/*/*/ckpt_epoch_*.pth
            python3 main.py $data -a $model --layers $layers -j 8 -C -g 0 -E --ckpt ckpt_best.pth
        done
    done
done

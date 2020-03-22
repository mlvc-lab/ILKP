#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2
    do
        python3 main.py $data -a $model -j 8 -C -g 0 1 2 3 -b 256 -N -v v1
        python3 main.py $data -a $model -j 8 -C -g 0 1 2 3 -b 256 -N -v v2
        rm -f checkpoint/*/*/ckpt_new_v*_epoch_*.pth
        python3 main.py $data -a $model -j 8 -C -g 0 -E -N --ckpt ckpt_new_v1_best.pth
        python3 main.py $data -a $model -j 8 -C -g 0 -E -N --ckpt ckpt_new_v2_best.pth
    done
done

#!/usr/bin/env bash

# for data in cifar10 cifar100 imagenet
for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    do
        if [ "$arch" = "resnet" ]; then
            echo "resnet"
            for layer in 20 32 44 56 110
            do
                python3 down_ckpt.py $data -a $model --layers $layer -o ckpt_best.pth
            done
        else
            echo "mobile friendly models"
            python3 down_ckpt.py $data -a $model -o ckpt_best.pth
        fi
    done
done

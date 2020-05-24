#!/usr/bin/env bash

# for model in mobilenet mobilenetv2 resnet resnext wideresnet vgg
for model in mobilenet mobilenetv2 resnet
do
    for data in cifar10 cifar100
    do
        if [ "$model" = "resnet" ]; then
            echo "resnet"
            for layer in 20 32 44 56 110
            do
                python3 down_ckpt.py $data -a $model --layers $layer -o ckpt_best.pth
            done
        else
            echo $model
            python3 down_ckpt.py $data -a $model -o ckpt_best.pth
        fi
    done
    if [ "$model" = "resnet" ]; then
        echo "resnet"
        for layer in 18 34 50 101 152
        do
            python3 down_ckpt.py imagenet -a $model --layers $layer -o ckpt_best.pth
        done
    else
        echo $model
        python3 down_ckpt.py imagenet -a $model -o ckpt_best.pth
    fi
done

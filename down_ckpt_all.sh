#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
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
done

for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
do
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

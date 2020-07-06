#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 resnet
    do
        if [ "$model" = "resnet" ]; then
            for layer in 20 32 44 56 110
            do
                # python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g 0 1 2 3 -N -v v2 -s 1
                python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g 0 1 -N -v v2 -s 1
                rm -f ./checkpoint/*/*/ckpt_new_v*_s*_epoch_*.pth
                python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g 0 1 -E -N --ckpt "ckpt_new_v2_s1_best.pth"
            done
        else
            # python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g 0 1 2 3 -N -v v2 -s 1
            python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g 0 1 -N -v v2 -s 1
            rm -f ./checkpoint/*/*/ckpt_new_v*_s*_epoch_*.pth
            python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g 0 1 -E -N --ckpt "ckpt_new_v2_pwd8_pws1_s1_best.pth"
        fi
    done
done

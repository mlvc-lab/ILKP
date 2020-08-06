#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 resnet
    do
        for wd in 1e-4 5e-4
        do
            if [ "$model" = "resnet" ]; then
                for layer in 20 32 44 56 110
                do
                    python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" --wd $wd --basetest
                    rm -f ./checkpoint/*/*/ckpt_wd*-*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_wd"$wd"_best.pth"
                done
            else
                python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" --wd $wd --basetest
                rm -f ./checkpoint/*/*/ckpt_wd*-*_epoch_*.pth
                python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_wd"$wd"_best.pth"
            fi
        done
    done
done

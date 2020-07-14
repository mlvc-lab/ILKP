#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 resnet
    do
        for qb in 8 7 6 5 4
        do
            if [ "$model" = "resnet" ]; then
                for layer in 20 32 44 56 110
                do
                    python3 quantize.py $data -a $model --layers $layer --ckpt ckpt_best.pth --qb $qb
                    python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_best_q"$qb".pth"
                done
            else
                python3 quantize.py $data -a $model --ckpt ckpt_best.pth --qb $qb
                python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_best_q"$qb".pth"
            fi
        done
    done
done

#!/usr/bin/env bash

for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv2 resnet
    for model in resnet
    do
        for qb in 8 6 4 3 2
        do
            if [ "$model" = "resnet" ]; then
                for layer in 20 32 44 56 110
                do
                    python3 quantize.py $data -a $model --layers $layer --ckpt ckpt_best.pth --qb $qb
                    python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_best_q"$qb".pth"
                    python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -b 256 --lr 0.01 -T --ckpt "ckpt_best_q"$qb".pth" -Q --qb $qb
                    rm -f ./checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_rt1_q"$qb"_best.pth"
                done
            else
                python3 quantize.py $data -a $model --ckpt ckpt_best.pth --qb $qb
                python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_best_q"$qb".pth"
                python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -b 256 --lr 0.01 -T --ckpt "ckpt_best_q"$qb".pth" -Q --qb $qb
                rm -f ./checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -E --ckpt "ckpt_rt1_q"$qb"_best.pth"
            fi
        done
    done
done

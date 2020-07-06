#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 resnet vgg
    do
        for qb in 8 6 4 3 2
        do
            if [ "$model" = "resnet" ]; then
                # for layer in 20 32 44 56 110
                for layer in 20 32 44
                do
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 1 2 3 -b 256 -N -v v2qq --qb $qb
                    rm -f checkpoint/*/*/ckpt_new_v*_q*_s*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E -N --ckpt "ckpt_new_v2qq_q"$qb"88_s5_best.pth" --qb $qb
                done
            else
                python3 main.py $data -a $model -j 2 -C -g 0 1 2 3 -b 256 -N -v v2qq --qb $qb
                rm -f checkpoint/*/*/ckpt_new_v*_q*_s*_epoch_*.pth
                python3 main.py $data -a $model -j 2 -C -g 0 -E -N --ckpt "ckpt_new_v2qq_pwd8_pws1_q"$qb"88_s5_best.pth" --qb $qb
            fi
        done
    done
done

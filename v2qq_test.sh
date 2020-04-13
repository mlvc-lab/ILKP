#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    do
        for qb in 8 6 4
        do
            if [ "$model" = "resnet" ]; then
                # for layer in 20 32 44 56 110
                for layer in 20 32 44
                do
                    python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2qq --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq_q"$qb"88.pth" --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq_q"$qb"88.pth" -N -v v2qq --qb $qb --lr 0.01
                    rm -f checkpoint/*/*/ckpt_rt*_v*_q*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_best.pth" --qb $qb
                done
            else
                python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2qq --qb $qb
                python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq_q"$qb"88.pth" --qb $qb
                python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq_q"$qb"88.pth" -N -v v2qq --qb $qb --lr 0.01
                rm -f checkpoint/*/*/ckpt_rt*_v*_q*_epoch_*.pth
                python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_best.pth" --qb $qb
            fi
        done
    done
done

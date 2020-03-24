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
                    python3 quantize.py $data -a $model --layer $layer --ckpt ckpt_best.pth --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E --ckpt ckpt_best_q$qb.pth
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 1 2 3 -T -Q --qb $qb -b 256 --ckpt ckpt_best_q$qb.pth --lr 0.01
                    rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_best.pth"
                done
            else
                python3 quantize.py $data -a $model --ckpt ckpt_best.pth --qb $qb
                python3 main.py $data -a $model -j 2 -C -g 0 -E --ckpt ckpt_best_q$qb.pth
                python3 main.py $data -a $model -j 2 -C -g 0 1 2 3 -T -Q --qb $qb -b 256 --ckpt ckpt_best_q$qb.pth --lr 0.01
                rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                python3 main.py $data -a $model -j 2 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_best.pth"
            fi
        done
    done
done

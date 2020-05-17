#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2
    do
        # for qb in 8 6 4 2
        for qb in 3
        do
            python3 quantize.py $data -a $model --ckpt ckpt_best.pth --qb $qb --pq -i
            python3 main.py $data -a $model -j 4 -C -g 0 -E --ckpt "ckpt_best_q"$qb"_pq_ifl.pth"
            python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T -Q --ckpt "ckpt_best_q"$qb"_pq_ifl.pth" --qb $qb --lr 0.01 --pq -i
            rm -f checkpoint/*/*/ckpt_rt*_q*_pq_ifl_epoch_*.pth
            python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -E --ckpt "ckpt_rt1_q"$qb"_pq_ifl_best.pth"
            python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2qqpq --qb $qb -i
            python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qqpq_q"$qb"88_ifl.pth" --qb $qb
            python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qqpq_q"$qb"88_ifl.pth" -N -v v2qqpq --qb $qb --lr 0.01 -i
            rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
            python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qqpq_q"$qb"88_ifl_best.pth" --qb $qb
        done
    done
done

#!/usr/bin/env bash

echo "quantization test"
for qb in 8 6 4 3 2
do
    for model in mobilenet mobilenetv2
    do
        echo "linear quantization baseline"
        python3 quantize.py imagenet -a $model --groups 3 --ckpt ckpt_best.pth --qb $qb
        python3 main.py imagenet -a $model --groups 3 -j 16 -C -g 0 1 -E --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet
        python3 main.py imagenet -a $model --groups 3 -j 16 -C -g 0 1 --wd 4e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb"_pq.pth" --datapath /dataset/ImageNet --lr 0.01 --epochs 90
        rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
        python3 main.py imagenet -a $model --groups 3 -j 16 -C -g 0 1 -E --ckpt "ckpt_rt1_q"$qb"_best.pth" --datapath /dataset/ImageNet
        echo "v2qq"
        python3 find_similar_kernel.py imagenet -a $model --groups 3 --ckpt ckpt_best.pth -v v2qq --qb $qb
        python3 main.py imagenet -a $model --groups 3 -j 16 -C -g 0 1 -E -N --ckpt "ckpt_best_v2qq_q"$qb"88.pth" --datapath /dataset/ImageNet --qb $qb
        python3 main.py imagenet -a $model --groups 3 -j 16 -C -g 0 1 --wd 4e-5 -p 500 -T --ckpt "ckpt_best_v2qq_q"$qb"88.pth" --datapath /dataset/ImageNet -N -v v2qq --qb $qb --lr 0.01 --epochs 90
        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_s*_epoch_*.pth
        python3 main.py imagenet -a $model --groups 3 -j 16 -C -g 0 1 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_s5_best.pth" --datapath /dataset/ImageNet --qb $qb
    done
done

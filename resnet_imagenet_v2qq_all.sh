#!/usr/bin/env bash

# for layer in 18 34 50 101 152
for layer in 18 34 50
do
    echo "v2qq test"
    for qb in 8 6 4 2
    do
        echo "linear quantization baseline"
        python3 quantize.py imagenet -a resnet --layer $layer --ckpt ckpt_best.pth --qb $qb
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 --wd 1e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet --lr 0.01
        rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E --ckpt "ckpt_rt1_q"$qb"_best.pth" --datapath /dataset/ImageNet
        echo "v2qq"
        python3 find_similar_kernel.py imagenet -a resnet --layers $layer --ckpt ckpt_best.pth -v v2qq --qb $qb
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E -N --ckpt "ckpt_best_v2qq_q"$qb"88.pth" --datapath /dataset/ImageNet --qb $qb
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 --wd 1e-5 -p 500 -T --ckpt "ckpt_best_v2qq_q"$qb"88.pth" --datapath /dataset/ImageNet -N -v v2qq --qb $qb --lr 0.01
        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_epoch_*.pth
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_best.pth" --datapath /dataset/ImageNet --qb $qb
    done
    echo "v2qq-ifl test"
    for qb in 8 6 4 2
    do
        echo "linear quantization-ifl baseline"
        python3 quantize.py imagenet -a resnet --layer $layer --ckpt ckpt_best.pth --qb $qb -i
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E --ckpt "ckpt_best_q"$qb"_ifl.pth" --datapath /dataset/ImageNet
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 --wd 1e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb"_ifl.pth" --datapath /dataset/ImageNet --lr 0.01 -i
        rm -f checkpoint/*/*/ckpt_rt*_q*_ifl_epoch_*.pth
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E --ckpt "ckpt_rt1_q"$qb"_ifl_best.pth" --datapath /dataset/ImageNet
        echo "v2qq-ifl"
        python3 find_similar_kernel.py imagenet -a resnet --layers $layer --ckpt ckpt_best.pth -v v2qq --qb $qb -i
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E -N --ckpt "ckpt_best_v2qq_q"$qb"88_ifl.pth" --datapath /dataset/ImageNet --qb $qb
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 --wd 1e-5 -p 500 -T --ckpt "ckpt_best_v2qq_q"$qb"88_ifl.pth" --datapath /dataset/ImageNet -N -v v2qq --qb $qb --lr 0.01 -i
        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
        python3 main.py imagenet -a resnet --layers $layer -j 16 -C -g 0 1 -b 256 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_ifl_best.pth" --datapath /dataset/ImageNet --qb $qb
    done
done

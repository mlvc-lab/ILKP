#!/usr/bin/env bash

for data in cifar10 cifar100
do
    echo "Original baseline training"
    python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 --lr 0.01
    python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -E --ckpt ckpt_best.pth
    echo "v2qq test"
    for qb in 8 6 4 2
    do
        echo "linear quantization baseline"
        python3 quantize.py $data -a vgg --layers 16 --ckpt ckpt_best.pth --qb $qb
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb".pth" --lr 0.001
        rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -E --ckpt "ckpt_rt1_q"$qb"_best.pth"
        echo "v2qq"
        python3 find_similar_kernel.py $data -a vgg --layers 16 --ckpt ckpt_best.pth -v v2qq --qb $qb
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq_q"$qb"88.pth" -N -v v2qq --qb $qb --lr 0.001
        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_epoch_*.pth
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_best.pth" --qb $qb
    done
    echo "v2qq-ifl test"
    for qb in 8 6 4 2
    do
        echo "linear quantization-ifl baseline"
        python3 quantize.py $data -a vgg --layers 16 --ckpt ckpt_best.pth --qb $qb -i
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb"_ifl.pth" --lr 0.001 -i
        rm -f checkpoint/*/*/ckpt_rt*_q*_ifl_epoch_*.pth
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -E --ckpt "ckpt_rt1_q"$qb"_ifl_best.pth"
        echo "v2qq-ifl"
        python3 find_similar_kernel.py $data -a vgg --layers 16 --ckpt ckpt_best.pth -v v2qq --qb $qb -i
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq_q"$qb"88_ifl.pth" -N -v v2qq --qb $qb --lr 0.001 -i
        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
        python3 main.py $data -a vgg --layers 16 -j 8 -C -g 0 1 -b 256 -E -N --ckpt "ckpt_rt1_v2qq_q"$qb"88_ifl_best.pth" --qb $qb
    done
done

#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 resnet
    do
        # for warm in 5 10 15 20
        for warm in 50
        do
            for scale in 1e-06 1e-05 1e-04
            do
                if [ "$model" = "resnet" ]; then
                    for layer in 20 32 44 56 110
                    do
                        # python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g 0 1 2 3 -N -v v2 -s 1 -warm $warm
                        python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -N -v v2 --tvl --tvls $scale -s 1 -warm $warm
                        rm -f ./checkpoint/*/*/ckpt_new_v*_tvl1e-*_s*_warm*_epoch_*.pth
                        python3 main.py $data -a $model --layers $layer --datapath /dataset/CIFAR -j 8 -C -g "$1" -E -N --ckpt "ckpt_new_v2_tvl"$scale"_s1_warm"$warm"_best.pth"
                    done
                else
                    # python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g 0 1 2 3 -N -v v2 -s 1 -warm $warm
                    python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -N -v v2 --tvl --tvls $scale -s 1 -warm $warm
                    rm -f ./checkpoint/*/*/ckpt_new_v*_tvl1e-*_s*_warm*_epoch_*.pth
                    python3 main.py $data -a $model --datapath /dataset/CIFAR -j 8 -C -g "$1" -E -N --ckpt "ckpt_new_v2_pwd8_pws1_tvl"$scale"_s1_warm"$warm"_best.pth"
                fi
            done
        done
    done
done

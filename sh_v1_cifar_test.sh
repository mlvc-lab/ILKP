#!/usr/bin/env bash

for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv1 resnet
    for model in resnet
    do
        if [ "$model" = "resnet" ]; then
            for layer in 20 32 44 56 110
            do
                python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v1
                python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -E -N --ckpt "ckpt_best_v1.pth"
                python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -b 256 -T --ckpt "ckpt_best_v1.pth" -N -v v1 --lr 0.01 --chk-save
                rm -f ./checkpoint/*/*/ckpt_rt*_v*_s*_epoch_*.pth
                python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -E -N --ckpt "ckpt_rt1_v1_s1_best.pth"
            done
        else
            python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v1
            python3 main.py $data -a $model -j 8 -C -g "$1" -E -N --ckpt "ckpt_best_v1_pwd8_pws1.pth"
            python3 main.py $data -a $model -j 8 -C -g "$1" -b 256 -T --ckpt "ckpt_best_v1_pwd8_pws1.pth" -N -v v1 --lr 0.01 --chk-save
            rm -f ./checkpoint/*/*/ckpt_rt*_v*_pwd8_pws1_s*_epoch_*.pth
            python3 main.py $data -a $model -j 8 -C -g "$1" -E -N --ckpt "ckpt_rt1_v1_pwd8_pws1_s1_best.pth"
        fi
    done
done

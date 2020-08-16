#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 resnet
    do
        if [ "$model" = "resnet" ]; then
            for layer in 20 32 44 56 110
            do
                python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2nb
                python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2nb.pth"
                python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2nb.pth" -N -v v2nb --lr 0.01
                rm -f checkpoint/*/*/ckpt_rt*_v*_s*_epoch_*.pth
                python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2nb_s1_best.pth"
            done
        else
            python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2nb
            python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2nb_pwd8_pws1.pth"
            python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2nb_pwd8_pws1.pth" -N -v v2nb --lr 0.01
            rm -f checkpoint/*/*/ckpt_rt*_v*_pwd8_pws1_s*_epoch_*.pth
            python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2nb_pwd8_pws1_s1_best.pth"
        fi
    done
done

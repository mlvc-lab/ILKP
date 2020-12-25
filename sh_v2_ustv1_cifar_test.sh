#!/usr/bin/env bash

for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv2 resnet
    for model in resnet
    do
        for ust in sigmoid tanh
        do
            if [ "$model" = "resnet" ]; then
                for layer in 20 32 44 56 110
                do
                    python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2 -ustv1 $ust
                    python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -E -N --ckpt "ckpt_best_v2_ustv1-"$ust".pth"
                    python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -b 256 -T --ckpt "ckpt_best_v2_ustv1-"$ust".pth" -N -v v2 -ustv1 $ust --lr 0.01
                    rm -f ./checkpoint/*/*/ckpt_rt*_v*_ustv1-*_s*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -E -N --ckpt "ckpt_rt1_v2_ustv1-"$ust"_s1_best.pth"
                done
            else
                python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2 -ustv1 $ust
                python3 main.py $data -a $model -j 8 -C -g "$1" -E -N --ckpt "ckpt_best_v2_pwd8_pws1_ustv1-"$ust".pth"
                python3 main.py $data -a $model -j 8 -C -g "$1" -b 256 -T --ckpt "ckpt_best_v2_pwd8_pws1_ustv1-"$ust".pth" -N -v v2 -ustv1 $ust --lr 0.01
                rm -f ./checkpoint/*/*/ckpt_rt*_v*_pwd8_pws1_ustv1-*_s*_epoch_*.pth
                python3 main.py $data -a $model -j 8 -C -g "$1" -E -N --ckpt "ckpt_rt1_v2_pwd8_pws1_ustv1-"$ust"_s1_best.pth"
            fi
        done
    done
done

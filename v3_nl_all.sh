#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    do
        if [ "$model" = "resnet" ]; then
            # for layer in 20 32 44 56 110
            for layer in 20 32 44
            do
                python3 find_similar_kernel.py $data -a $model --layer $layer --ckpt ckpt_best.pth -v v3 -d 2
                python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E -N --ckpt ckpt_best_v3_d2.pth -d 2
            done
        else
            python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v3 -d 2
            python3 main.py $data -a $model -j 2 -C -g 0 -E -N --ckpt ckpt_best_v3_d2.pth -d 2
        fi
    done
done

for data in cifar10 cifar100
do
    for nls in 1.0 0.1 0.01
    do
        for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
        do
            if [ "$model" = "resnet" ]; then
                # for layer in 20 32 44 56 110
                for layer in 20 32 44
                do
                    python3 main.py $data -a $model --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt ckpt_best_v3_d2.pth -N -v v3 -d 2 --lr 0.01 --nl --nls $nls
                    rm -f checkpoint/*/*/ckpt_rt*_v3_nl*_s*_d*_epoch_*.pth
                    python3 main.py $data -a $model --layer $layer -j 8 -C -g 0 -E -N --ckpt "ckpt_rt1_v3_nl"$nls"_s5_d2_best.pth" -d 2
                done
            else
                python3 main.py $data -a $model -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt ckpt_best_v3_d2.pth -N -v v3 -d 2 --lr 0.01 --nl --nls $nls
                rm -f checkpoint/*/*/ckpt_rt*_v3_nl*_s*_d*_epoch_*.pth
                python3 main.py $data -a $model -j 8 -C -g 0 -E -N --ckpt "ckpt_rt1_v3_nl"$nls"_s5_d2_best.pth" -d 2
            fi
        done
    done
done

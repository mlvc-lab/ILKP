#!/usr/bin/env bash

for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv2 resnet
    for model in resnet
    do
        for scale in 1e-03 1e-04 1e-05 1e-06 1e-07
        do
            for group in 1 2 4 8 16
            do
                if [ "$model" = "resnet" ]; then
                    for layer in 20 32 44 56 110
                    do
                        # python3 main.py $data -a $model --layers $layer -j 8 -C -g 0 1 2 3
                        python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" --groupcorl --groupcorls $scale --groupcorn $group
                        rm -f ./checkpoint/*/*/ckpt_groupcorl1e-*_g*_epoch_*.pth
                        python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -E --ckpt "ckpt_groupcorl"$scale"_g"$group"_best.pth"
                        python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt "ckpt_groupcorl"$scale"_g"$group"_best.pth" -v v2
                        python3 main.py $data -a $model --layers $layer -j 8 -C -g "$1" -E -N --ckpt "ckpt_groupcorl"$scale"_g"$group"_best_v2.pth"
                    done
                else
                    # python3 main.py $data -a $model -j 8 -C -g 0 1 2 3
                    python3 main.py $data -a $model -j 8 -C -g "$1" --groupcorl --groupcorls $scale --groupcorn $group
                    rm -f ./checkpoint/*/*/ckpt_groupcorl1e-*_g*_epoch_*.pth
                    python3 main.py $data -a $model -j 8 -C -g "$1" -E --ckpt "ckpt_groupcorl"$scale"_g"$group"_best.pth"
                    python3 find_similar_kernel.py $data -a $model --ckpt "ckpt_groupcorl"$scale"_g"$group"_best.pth" -v v2
                    python3 main.py $data -a $model -j 8 -C -g "$1" -E -N --ckpt "ckpt_groupcorl"$scale"_g"$group"_best_v2.pth"
                fi
            done
        done
    done
done

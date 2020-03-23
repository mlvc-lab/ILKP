#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    do
        for ver in v1 v2
        do
            if [ "$model" = "resnet" ]; then
                for layer in 20 32 44 56 110
                do
                    python3 main.py $data -a $model --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt "ckpt_best_"$ver".pth" -N -v $ver --lr 0.01
                    rm -f checkpoint/*/*/ckpt_rt*_v*_epoch_*.pth
                    python3 main.py $data -a $model --layer $layer -j 8 -C -g 0 -E -N --ckpt "ckpt_rt1_"$ver"_best.pth"
                done
            else
                python3 main.py $data -a $model -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt "ckpt_best_"$ver".pth" -N -v $ver --lr 0.01
                rm -f checkpoint/*/*/ckpt_rt*_v*_epoch_*.pth
                python3 main.py $data -a $model -j 8 -C -g 0 -E -N --ckpt "ckpt_rt1_"$ver"_best.pth"
            fi
        done
    done
done

# for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
# do
#     for ver in v1 v2
#     do
#         if [ "$model" = "resnet" ]; then
#             for layer in 18 34 50 101 152
#             do
#                 python3 main.py imagenet -a $model --layer $layer -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt "ckpt_best_"$ver".pth" -N -v $ver --lr 0.01
#                 rm -f checkpoint/*/*/ckpt_rt*_v*_epoch_*.pth
#                 python3 main.py imagenet -a $model --layer $layer -j 8 -C -g 0 -E -N --ckpt "ckpt_rt1_"$ver"_best.pth"
#             done
#         else
#             python3 main.py imagenet -a $model -j 8 -C -g 0 1 2 3 -b 256 -T --ckpt "ckpt_best_"$ver".pth" -N -v $ver --lr 0.01
#             rm -f checkpoint/*/*/ckpt_rt*_v*_epoch_*.pth
#             python3 main.py imagenet -a $model -j 8 -C -g 0 -E -N --ckpt "ckpt_rt1_"$ver"_best.pth"
#         fi
#     done
# done

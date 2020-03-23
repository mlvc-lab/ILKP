#!/usr/bin/env bash

for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    do
        if [ "$model" = "resnet" ]; then
            for layer in 20 32 44 56 110
            do
                python3 main.py $data -a $model --layer $layer -j 8 -C -g 0 1 2 3 -b 256
                rm -f checkpoint/*/*/ckpt_epoch_*.pth
                python3 main.py $data -a $model --layer $layer -j 8 -C -g 0 -E --ckpt ckpt_best.pth
            done
        else
            python3 main.py $data -a $model -j 8 -C -g 0 1 2 3 -b 256
            rm -f checkpoint/*/*/ckpt_epoch_*.pth
            python3 main.py $data -a $model -j 8 -C -g 0 -E --ckpt ckpt_best.pth
        fi
    done
done

# for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
# do
#     if [ "$model" = "resnet" ]; then
#         for layer in 18 34 50 101 152
#         do
#             python3 main.py imagenet -a $model --layer $layer -j 8 -C -g 0 1 2 3 -b 256
#             rm -f checkpoint/*/*/ckpt_epoch_*.pth
#             python3 main.py imagenet -a $model --layer $layer -j 8 -C -g 0 -E --ckpt ckpt_best.pth
#         done
#     else
#         python3 main.py imagenet -a $model -j 8 -C -g 0 1 2 3 -b 256
#         rm -f checkpoint/*/*/ckpt_epoch_*.pth
#         python3 main.py imagenet -a $model -j 8 -C -g 0 -E --ckpt ckpt_best.pth
#     fi
# done

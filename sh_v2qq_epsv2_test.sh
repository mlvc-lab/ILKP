#!/usr/bin/env bash

echo "v2qq-epsv2"
for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv2 resnet vgg
    for model in mobilenet mobilenetv2 resnet
    do
        # for qb in 8 6 4 3 2
        for qb in 6
        do
            for eps in 1e-05 1e-06 1e-07 1e-08 1e-09
                if [ "$model" = "resnet" ]; then
                    # for layer in 20 32 44 56 110
                    for layer in 20 32 44
                    do
                        python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2qq-epsv2 --qb $qb -eps $eps
                        python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq-epsv2_q"$qb"88_eps"$eps".pth" --qb $qb
                        python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq-epsv2_q"$qb"88_eps"$eps".pth" -N -v v2qq-epsv2 --qb $qb --lr 0.01 -eps $eps
                        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_eps1e-*_s*_epoch_*.pth
                        python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq-epsv2_q"$qb"88_eps"$eps"_s5_best.pth" --qb $qb
                    done
                else
                    python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2qq-epsv2 --qb $qb -eps $eps
                    python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq-epsv2_pwd8_pws1_q"$qb"88_eps"$eps".pth" --qb $qb
                    python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq-epsv2_pwd8_pws1_q"$qb"88_eps"$eps".pth" -N -v v2qq-epsv2 --qb $qb --lr 0.01 -eps $eps
                    rm -f checkpoint/*/*/ckpt_rt*_v*_pwd8_pws1_q*_eps1e-*_s*_epoch_*.pth
                    python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq-epsv2_pwd8_pws1_q"$qb"88_eps"$eps"_s5_best.pth" --qb $qb
                fi
            done
        done
    done
done

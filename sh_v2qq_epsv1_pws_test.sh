#!/usr/bin/env bash

echo "v2qq-epsv1-pws_test"
for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2
    do
        # for qb in 8 6 4 3 2
        for qb in 6
        do
            for pws in 1 2 4 8
            do
                python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2qq-epsv1 --qb $qb -pws $pws
                python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq-epsv1_pwd8_pws"$pws"_q"$qb"88_eps1e-08.pth" --qb $qb -pws $pws
                python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq-epsv1_pwd8_pws"$pws"_q"$qb"88_eps1e-08.pth" -N -v v2qq-epsv1 --qb $qb --lr 0.01 -pws $pws
                rm -f checkpoint/*/*/ckpt_rt*_v*_pwd8_pws*_q*_eps1e-08_s*_epoch_*.pth
                python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq-epsv1_pwd8_pws"$pws"_q"$qb"88_eps1e-08_s5_best.pth" --qb $qb -pws $pws
            done
        done
    done
done

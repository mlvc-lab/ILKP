#!/usr/bin/env bash

for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv2 resnet
    for model in resnet
    do
        # for qba in 8 7 6 5 4 3 2
        for qba in 8
        do
            # for qbb in 8 7 6 5 4 3 2
            for qbb in 8
            do
                if [ "$model" = "resnet" ]; then
                    for layer in 20 32 44 56 110
                    # for layer in 20 32 44
                    do
                        echo "v2q"
                        python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2q --qba $qba --qbb $qbb
                        python3 main.py $data -a $model --layers $layer -j 4 -C -g "$1" -E -N --ckpt "ckpt_best_v2q_qa"$qba"b"$qbb"_eps1e-08.pth" --qba $qba --qbb $qbb
                        python3 main.py $data -a $model --layers $layer -j 4 -C -g "$1" -b 256 -T --ckpt "ckpt_best_v2q_qa"$qba"b"$qbb"_eps1e-08.pth" -N -v v2q --qba $qba --qbb $qbb --lr 0.01
                        rm -f checkpoint/*/*/ckpt_rt*_v*_q*_eps1e-08_s*_epoch_*.pth
                        python3 main.py $data -a $model --layers $layer -j 4 -C -g "$1" -E -N --ckpt "ckpt_rt1_v2q_qa"$qba"b"$qbb"_eps1e-08_s1_best.pth" --qba $qba --qbb $qbb
                    done
                else
                    echo "v2q"
                    python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2q --qba $qba --qbb $qbb
                    python3 main.py $data -a $model -j 4 -C -g "$1" -E -N --ckpt "ckpt_best_v2q_pwd8_pws1_qa"$qba"b"$qbb"_eps1e-08.pth" --qba $qba --qbb $qbb
                    python3 main.py $data -a $model -j 4 -C -g "$1" -b 256 -T --ckpt "ckpt_best_v2q_pwd8_pws1_qa"$qba"b"$qbb"_eps1e-08.pth" -N -v v2q --qba $qba --qbb $qbb --lr 0.01
                    rm -f checkpoint/*/*/ckpt_rt*_v*_pwd8_pws1_q*_eps1e-08_s*_epoch_*.pth
                    python3 main.py $data -a $model -j 4 -C -g "$1" -E -N --ckpt "ckpt_rt1_v2q_pwd8_pws1_qa"$qba"b"$qbb"_eps1e-08_s1_best.pth" --qba $qba --qbb $qbb
                fi
            done
        done
    done
done

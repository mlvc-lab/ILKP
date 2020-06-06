#!/usr/bin/env bash

for data in cifar10 cifar100
do
    # for model in mobilenet mobilenetv2 resnet vgg
    for model in mobilenet mobilenetv2 resnet
    do
        for qb in 8 6 4 3 2
        do
            if [ "$model" = "resnet" ]; then
                # for layer in 20 32 44 56 110
                for layer in 20 32 44
                do
                    # echo "linear quantization baseline"
                    # python3 quantize.py $data -a $model --layers $layer --ckpt ckpt_best.pth --qb $qb
                    # python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E --ckpt "ckpt_best_q"$qb".pth" --qb $qb
                    # python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 1 -b 256 -T -Q --ckpt "ckpt_best_q"$qb".pth" --qb $qb --lr 0.01
                    # rm -rf checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                    # python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_best.pth" --qb $qb
                    echo "v2qq-epsv1"
                    python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2qq-epsv1 --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq-epsv1_q"$qb"88_eps1e-08.pth" --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq-epsv1_q"$qb"88_eps1e-08.pth" -N -v v2qq-epsv1 --qb $qb --lr 0.01
                    rm -f checkpoint/*/*/ckpt_rt*_v*_q*_eps1e-08_s*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq-epsv1_q"$qb"88_eps1e-08_s5_best.pth" --qb $qb
                done
            else
                # echo "linear quantization baseline"
                # python3 quantize.py $data -a $model --ckpt ckpt_best.pth --qb $qb
                # python3 main.py $data -a $model -j 4 -C -g 0 -E --ckpt "ckpt_best_q"$qb".pth" --qb $qb
                # python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T -Q --ckpt "ckpt_best_q"$qb".pth" --qb $qb --lr 0.01
                # rm -rf checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                # python3 main.py $data -a $model -j 4 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_best.pth" --qb $qb
                echo "v2qq-epsv1"
                python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2qq-epsv1 --qb $qb
                python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_best_v2qq-epsv1_q"$qb"88_eps1e-08.pth" --qb $qb
                python3 main.py $data -a $model -j 4 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2qq-epsv1_q"$qb"88_eps1e-08.pth" -N -v v2qq-epsv1 --qb $qb --lr 0.01
                rm -f checkpoint/*/*/ckpt_rt*_v*_q*_eps1e-08_s*_epoch_*.pth
                python3 main.py $data -a $model -j 4 -C -g 0 -E -N --ckpt "ckpt_rt1_v2qq-epsv1_q"$qb"88_eps1e-08_s5_best.pth" --qb $qb
            fi
        done
    done
done

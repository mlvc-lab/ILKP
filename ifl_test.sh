#!/usr/bin/env bash

echo "quantization-ifl test"
for data in cifar10 cifar100
do
    for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    do
        for qb in 8 6 4
        do
            if [ "$model" = "resnet" ]; then
                # for layer in 20 32 44 56 110
                for layer in 20 32 44
                do
                    echo "linear quantization-ifl baseline"
                    python3 quantize.py $data -a $model --layer $layer --ckpt ckpt_best.pth --qb $qb -i
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E --ckpt "ckpt_best_q"$qb"_ifl.pth"
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 1 -T -Q --qb $qb -b 256 --ckpt "ckpt_best_q"$qb"_ifl.pth" --lr 0.01 -i
                    rm -f checkpoint/*/*/ckpt_rt*_q*_ifl_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_ifl_best.pth"
                    echo "v2q-ifl"
                    python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2q --qb $qb -i
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E -N --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" -N -v v2q --qb $qb --lr 0.01 -i
                    rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 2 -C -g 0 -E -N --ckpt "ckpt_rt1_v2q_q"$qb"_ifl_best.pth" --qb $qb
                done
            else
                echo "linear quantization-ifl baseline"
                python3 quantize.py $data -a $model --ckpt ckpt_best.pth --qb $qb -i
                python3 main.py $data -a $model -j 2 -C -g 0 -E --ckpt "ckpt_best_q"$qb"_ifl.pth"
                python3 main.py $data -a $model -j 2 -C -g 0 1 -T -Q --qb $qb -b 256 --ckpt "ckpt_best_q"$qb"_ifl.pth" --lr 0.01 -i
                rm -f checkpoint/*/*/ckpt_rt*_q*_ifl_epoch_*.pth
                python3 main.py $data -a $model -j 2 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_ifl_best.pth"
                echo "v2q-ifl"
                python3 find_similar_kernel.py $data -a $model --ckpt ckpt_best.pth -v v2q --qb $qb -i
                python3 main.py $data -a $model -j 2 -C -g 0 -E -N --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" --qb $qb
                python3 main.py $data -a $model -j 2 -C -g 0 1 -b 256 -T --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" -N -v v2q --qb $qb --lr 0.01 -i
                rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
                python3 main.py $data -a $model -j 2 -C -g 0 -E -N --ckpt "ckpt_rt1_v2q_q"$qb"_best.pth" --qb $qb
            fi
        done
    done
done

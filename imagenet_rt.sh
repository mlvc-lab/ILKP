#!/usr/bin/env bash

echo "quantization-ifl test"
for qb in 8 6 4 2
do
    # for model in mobilenet mobilenetv2 shufflenet shufflenetv2 resnet
    for model in mobilenet mobilenetv2 shufflenetv2 resnet
    do
        for data in imagenet
        do
            if [ "$model" = "resnet" ]; then
                # for layer in 18 34 50 101 152
                for layer in 50
                do
                    echo "linear quantization baseline"
                    python3 quantize.py $data -a $model --layer $layer --ckpt ckpt_best.pth --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 -E --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 --wd 1e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet --lr 0.01
                    rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 -E --ckpt "ckpt_rt1_q"$qb"_best.pth" --datapath /dataset/ImageNet
                    echo "linear quantization-ifl baseline"
                    python3 quantize.py $data -a $model --layer $layer --ckpt ckpt_best.pth --qb $qb -i
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 -E --ckpt "ckpt_best_q"$qb"_ifl.pth" --datapath /dataset/ImageNet
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 --wd 1e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb"_ifl.pth" --datapath /dataset/ImageNet --lr 0.01 -i
                    rm -f checkpoint/*/*/ckpt_rt*_q*_ifl_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 -E --ckpt "ckpt_rt1_q"$qb"_ifl_best.pth" --datapath /dataset/ImageNet
                    echo "v2q-ifl"
                    python3 find_similar_kernel.py $data -a $model --layers $layer --ckpt ckpt_best.pth -v v2q --qb $qb -i
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 -E -N --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" --datapath /dataset/ImageNet --qb $qb
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 --wd 1e-5 -p 500 -T --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" --datapath /dataset/ImageNet -N -v v2q --qb $qb --lr 0.01 -i
                    rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
                    python3 main.py $data -a $model --layers $layer -j 16 -C -g 0 1 -E -N --ckpt "ckpt_rt1_v2q_q"$qb"_ifl_best.pth" --datapath /dataset/ImageNet --qb $qb
                done
            else
                echo "linear quantization baseline"
                python3 quantize.py $data -a $model --groups 3 --ckpt ckpt_best.pth --qb $qb
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 -E --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 --wd 4e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb".pth" --datapath /dataset/ImageNet --lr 0.01
                rm -f checkpoint/*/*/ckpt_rt*_q*_epoch_*.pth
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 -E --ckpt "ckpt_rt1_q"$qb"_best.pth" --datapath /dataset/ImageNet
                echo "linear quantization-ifl baseline"
                python3 quantize.py $data -a $model --groups 3 --ckpt ckpt_best.pth --qb $qb -i
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 -E --ckpt "ckpt_best_q"$qb"_ifl.pth" --datapath /dataset/ImageNet
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 --wd 4e-5 -p 500 -T -Q --qb $qb --ckpt "ckpt_best_q"$qb"_ifl.pth" --datapath /dataset/ImageNet --lr 0.01 -i
                rm -f checkpoint/*/*/ckpt_rt*_q*_ifl_epoch_*.pth
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 -E --ckpt "ckpt_rt1_q"$qb"_ifl_best.pth" --datapath /dataset/ImageNet
                echo "v2q-ifl"
                python3 find_similar_kernel.py $data -a $model --groups 3 --ckpt ckpt_best.pth -v v2q --qb $qb -i
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 -E -N --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" --datapath /dataset/ImageNet --qb $qb
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 --wd 4e-5 -p 500 -T --ckpt "ckpt_best_v2q_q"$qb"_ifl.pth" --datapath /dataset/ImageNet -N -v v2q --qb $qb --lr 0.01 -i
                rm -f checkpoint/*/*/ckpt_rt*_v*_q*_ifl_epoch_*.pth
                python3 main.py $data -a $model --groups 3 -j 16 -C -g 0 1 -E -N --ckpt "ckpt_rt1_v2q_q"$qb"_ifl_best.pth" --datapath /dataset/ImageNet --qb $qb
            fi
        done
    done
done

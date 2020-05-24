# Memory Efficient Storing Scheme

For my research..
You can train or test MobileNet/MobileNetV2/ResNet/VGG on CIFAR10/CIFAR100/ImageNet.  
Specially, you can train or test on any device (CPU/sinlge GPU/multi GPU) and resume on different device environment available.

## Requirements

- `python 3.5+`
- `pytorch 1.0+`
- `torchvision 0.4+`
- `numpy`
- `tqdm`
- `requests` (for downloading pretrained checkpoint and imagenet dataset)

----------

## Details of storing version

- v2
  - v2
  - v2q (quantization ver)
  - v2qq (quantization ver with alpha, beta quantization)
  - v2f (fixed index $k$ during retraining time with v2qq)
  - v2nb (no $\beta$ with v2qq)
    - np: no adaptation v2, v2q, ⋯ for pointwise convolutional layers

----------

## TODO

- Update other models
  - VGG pretrained model
  - ResNext, WideResNet coding
- 여러가지 flag -> rotation, shift 추가해서 성능확인
- Make TinyImageNet dataloader

----------

## Number of (Depth-wise) Convolution kernels in Models

### Mobile-friendly models - CIFAR - depth-wise convolution

| LayerNum |  MobileNet   | MobileNetV2 |
|:--------:|:------------:|:-----------:|
|    0     | (3X3X1)X32   | (3X3X1)X32  |
|    1     | (3X3X1)X64   | (3X3X1)X96  |
|    2     | (3X3X1)X128  | (3X3X1)X144 |
|    3     | (3X3X1)X128  | (3X3X1)X144 |
|    4     | (3X3X1)X256  | (3X3X1)X192 |
|    5     | (3X3X1)X256  | (3X3X1)X192 |
|    6     | (3X3X1)X512  | (3X3X1)X192 |
|    7     | (3X3X1)X512  | (3X3X1)X384 |
|    8     | (3X3X1)X512  | (3X3X1)X384 |
|    9     | (3X3X1)X512  | (3X3X1)X384 |
|    10    | (3X3X1)X512  | (3X3X1)X384 |
|    11    | (3X3X1)X512  | (3X3X1)X576 |
|    12    | (3X3X1)X1024 | (3X3X1)X576 |
|    13    |              | (3X3X1)X576 |
|    14    |              | (3X3X1)X960 |
|    15    |              | (3X3X1)X960 |
|    16    |              | (3X3X1)X960 |
|    17    |              |             |
|    18    |              |             |

### ResNets - CIFAR - convolution

| LayerNum |  ResNet20  | \| | LayerNum |  ResNet32  | \| | LayerNum |  ResNet44  | \| | LayerNum |  ResNet56  | \| | LayerNum |  ResNet110  |
|:--------:|:----------:|:--:|:--------:|:----------:|:--:|:--------:|:----------:|:--:|:--------:|:----------:|:--:|:--------:|:-----------:|
|  0      | (3X3X3)X16  | \| |  0      | (3X3X3)X16  | \| |  0      | (3X3X3)X16  | \| |  0      | (3X3X3)X16  | \| |  0       | (3X3X3)X16  |
|  1~6    | (3X3X16)X16 | \| |  1~10   | (3X3X16)X16 | \| |  1~14   | (3X3X16)X16 | \| |  1~18   | (3X3X16)X16 | \| |  1~36    | (3X3X16)X16 |
|  7      | (3X3X16)X32 | \| |  11     | (3X3X16)X32 | \| |  15     | (3X3X16)X32 | \| |  19     | (3X3X16)X32 | \| |  37      | (3X3X16)X32 |
|  8~12   | (3X3X32)X32 | \| |  12~20  | (3X3X32)X32 | \| |  16~28  | (3X3X32)X32 | \| |  20~36  | (3X3X32)X32 | \| |  38~72   | (3X3X32)X32 |
|  13     | (3X3X32)X64 | \| |  21     | (3X3X32)X64 | \| |  29     | (3X3X32)X64 | \| |  37     | (3X3X32)X64 | \| |  73      | (3X3X32)X64 |
|  14~18  | (3X3X64)X64 | \| |  22~30  | (3X3X64)X64 | \| |  30~42  | (3X3X64)X64 | \| |  38~54  | (3X3X64)X64 | \| |  74~108  | (3X3X64)X64 |

### Others

Soon...

----------

## How to download the ImageNet data

```
usage: down_imagenet.py [-h] [--datapath PATH]

optional arguments:
  -h, --help       show this help message and exit
  --datapath PATH  Where you want to save ImageNet? (default: ../data)
```

### usage

``` shell
$ python3 down_imagenet.py
```

> ***Please check the datapath***  
> Match the same as the datapath argument used by **`main.py`**.

## How to download a pretrained Model

The pretrained models of ShuffleNet, ShuffleNetV2 trained on ImegeNet is not available now..  
All the checkpoint files of ResNets trained on ImageNet are from the official torchvision models.  
So, if you use that checkpoints, you can't resume right condition..  
But, you can retrain or apply the MESS using those checkpoints.

### Usage

``` shell
$ python down_ckpt.py imagenet -a mobilenet -o pretrained_model.pth
```

***for downloading all checkpoints***

``` shell
$ ./down_ckpt_all.sh
```

----------

## How to train / test networks

```
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [-b N] [--lr LR]
               [--momentum M] [--wd W] [--layers N] [--bn] [--width-mult WM]
               [--groups N] [-p N] [--ckpt PATH] [-R] [-E] [-C] [-T]
               [-g GPUIDS [GPUIDS ...]] [--datapath PATH] [-v V] [-d N]
               [-pwd N] [-N] [-s N] [--nl] [--nls NLS] [--pl] [--pls PLS] [-Q]
               [--np] [--qb N] [--qba N] [--qbb N]
               DATA

KH Research

positional arguments:
  DATA                  dataset: cifar10 | cifar100 | imagenet (default:
                        cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mobilenet | mobilenetv2 | resnet |
                        vgg (default: mobilenet)
  -j N, --workers N     number of data loading workers (default: 8)
  --epochs N            number of total epochs to run (default: 200)
  -b N, --batch-size N  mini-batch size (default: 128), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate (defualt: 0.1)
  --momentum M          momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 5e-4)
  --layers N            number of layers in VGG/ResNet/ResNeXt/WideResNet
                        (default: 16)
  --bn, --batch-norm    Use batch norm in VGG?
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  --groups N            number of groups for ShuffleNet (default: 2)
  -p N, --print-freq N  print frequency (default: 100)
  --ckpt PATH           Path of checkpoint for resuming/testing or retraining
                        model (Default: none)
  -R, --resume          Resume model?
  -E, --evaluate        Test model?
  -C, --cuda            Use cuda?
  -T, --retrain         Retraining?
  -g GPUIDS [GPUIDS ...], --gpuids GPUIDS [GPUIDS ...]
                        GPU IDs for using (Default: 0)
  --datapath PATH       where you want to load/save your dataset? (default:
                        ../data)
  -v V, --version V     version: v2 | v2q | v2qq | v2qpq | v2qqpq | v2f | v2nb
                        (find kernel version (default: none))
  -d N, --bind-size N   the number of binding channels in convolution
                        (subvector size) on version 3 (v3) (default: 2)
  -pwd N, --pw-bind-size N
                        the number of binding channels in pointwise
                        convolution (subvector size) (default: 8)
  -N, --new             new method?
  -s N, --save-epoch N  number of epochs to save checkpoint and to apply new
                        method
  --nl, --nuc-loss      nuclear norm loss?
  --nls NLS, --nl-scale NLS
                        scale factor of nuc_loss
  --pl, --pcc-loss      pearson correlation coefficient loss?
  --pls PLS, --pl-scale PLS
                        scale factor of pcc_loss
  -Q, --quant           use quantization?
  --np                  no v2-like method in pointwise convolutional layer?
  --qb N, --quant-bit N
                        number of bits for quantization (Default: 8)
  --qba N, --quant_bit_a N
                        number of bits for quantizing alphas (Default: 8)
  --qbb N, --quant_bit_b N
                        number of bits for quantizing betas (Default: 8)
```

### Training

#### Train one network with a certain dataset

``` shell
$ python main.py cifar10 -a mobilenet -C -g 0 1 2 3 -b 256
```

#### Resume training

``` shell
$ python main.py cifar10 -a mobilenet -C -g 0 1 2 3 -b 256 -R --ckpt ckpt_epoch_50.pth
```

### Test

``` shell
$ python main.py cifar10 -a mobilenet -C -g 0 1 2 3 -b 256 -E --ckpt ckpt_best.pth
```

## Delete Checkpoints (without best validation accuracy checkpoint)

``` shell
$ rm -f checkpoint/*/*/ckpt_epoch_*.pth
```

----------

## References

- [torchvision models github codes](https://github.com/pytorch/vision/tree/master/torchvision/models)
- [MobileNet, ShuffleNet and ShuffleNetV2 Cifar GitHub (unofficial)](https://github.com/kuangliu/pytorch-cifar)
- [MobileNetV2 Cifar GitHub (unofficial)](https://github.com/tinyalpha/mobileNet-v2_cifar10)
- [ShuffleNet and ShuffleNetV2 GitHub (unofficial)](https://github.com/xingmimfl/pytorch_ShuffleNet_ShuffleNetV2)
- [ShuffleNet GitHub (unofficial)](https://github.com/jaxony/ShuffleNet)
- [ShuffleNetV2 GitHub (unofficial)](https://github.com/Randl/ShuffleNetV2-pytorch)
- [PyTorch-CIFAR100 Benchmark list](https://github.com/weiaicunzai/pytorch-cifar100)
- [VGG-CIFAR GitHub (unofficial)](https://github.com/chengyangfu/pytorch-vgg-cifar10)

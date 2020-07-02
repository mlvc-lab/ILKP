# Memory Efficient Storing Scheme

For my research..  
You can train or test MobileNet/MobileNetV2/ResNet/VGG on CIFAR10/CIFAR100/ImageNet.  
Specially, you can train or test on any device (CPU/sinlge GPU/multi GPU) and resume on different device environment available.

----------

## Requirements

- `python 3.5+`
- `pytorch 1.0+`
- `torchvision 0.4+`
- `numpy`
- `tqdm`
- `rich` (for beautiful log on console) (not yet)
- `requests` (for downloading pretrained checkpoint and imagenet dataset)
- `sacred` (for logging on omniboard)
- `pymongo` (for logging on omniboard)

----------

## Details of storing version

- v2
  - v2
  - v2q (quantization ver)
  - v2qq (quantization ver with alpha, beta quantization)
    - v2qq-epsv1 (add epsilon to every denom with v2qq)
    - v2qq-epsv2 (if denom is 0, set denom to epsilon with v2qq)
    - v2qq-epsv3 (if alpha is nan, set alpha to 1.0 with v2qq)
  - v2f (fixed index $k$ during retraining time with v2qq)
  - v2nb (no $\beta$ with v2qq)
    - np: no adaptation v2, v2q, ⋯ for pointwise convolutional layers
  - v2.5 (old: v3) (binding dwkernels) (현재는 사용 불가)
- v3: rotation, flip, shift (구현중지)

----------

## TODO

- V2 Full precision(32bit)에서 나온 결과들 확인
- quantization 안하고 fine-tuning도 해보기
- fine-tuning없이 처음부터 학습하는거 결과 다시 보고
- 처음부터 학습할 때 loss term 추가 혹은 과도한 outlier weight를 제거 및 regularization을 위한 weight clipping
- Update other models
  - VGG pretrained model upload on google drive
  - WideResNet coding
- 여러가지 flag -> rotation, shift 추가해서 성능확인 (v3)
- Make TinyImageNet dataloader

----------

## Files

- `check_model_params.py`: optional file for calculating number of parameters
- `config.py`: set configuration
- `data.py`: data loading
- `down_ckpt.py`: download checkpoints of pretrained models
- `down_ckpt_all.sh`: shell file of downloading all checkpoints of pretrained models
- `down_imagenet.py`: download the ImageNet dataset (ILSVRC2012 ver.)
- `find_similar_kernel.py`: find similar kernel
- `main.py`: main python file for training or testing
- `models`
  - `__init__.py`
  - `mobilenet.py`
  - `mobilenetv2.py`
  - `resnet.py`
  - `vgg.py`
- `quantize.py`
- `sh_mobile_imagenet_v2qq.sh`
- `sh_resnet_imagenet_v2qq.sh`
- `sh_v2nb_cifar_test.sh`
- `sh_v2qq_cifar_run.sh`
- `sh_v2qq_cifar_test.sh`
- `sh_vgg16_cifar_v2qq.sh`
- `torchvision_to_ours.py`: change checkpoints files from pretrained models in torchvision to our states
- `utils.py`

----------

## How to download the ImageNet data

``` text
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

The pretrained models of VGG trained on ImegeNet is not available now..  
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

``` text
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [-b N] [--lr LR]
               [--momentum M] [--wd W] [--layers N] [--bn] [--width-mult WM]
               [--groups N] [-p N] [--ckpt PATH] [-R] [-E] [-C] [-T]
               [-g GPUIDS [GPUIDS ...]] [--datapath PATH] [-v V] [-d N]
               [-pwd N] [-N] [-s N] [--nl] [--nls NLS] [--pl] [--pls PLS] [-Q]
               [--np] [--qb N] [--qba N] [--qbb N]
               DATA

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
  --layers N            number of layers in VGG/ResNet/WideResNet
                        (default: 16)
  --bn, --batch-norm    Use batch norm in VGG?
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
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
  -v V, --version V     version: v2 | v2q | v2qq | v2f | v2nb
                        (find kernel version (default: none))
  -d N, --bind-size N   the number of binding channels in convolution
                        (subvector size) on version 2.5 (v2.5) (default: 2)
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
- [MobileNet Cifar GitHub (unofficial)](https://github.com/kuangliu/pytorch-cifar)
- [MobileNetV2 Cifar GitHub (unofficial)](https://github.com/tinyalpha/mobileNet-v2_cifar10)
- [PyTorch-CIFAR100 Benchmark list](https://github.com/weiaicunzai/pytorch-cifar100)
- [VGG-CIFAR GitHub (unofficial)](https://github.com/chengyangfu/pytorch-vgg-cifar10)

import argparse
import models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

r'''version list
- v2
  - v2
  - v2q (quantization ver)
  - v2qq (quantization ver with alpha, beta qunatization)
  - v2qpq (v2q with pwconv quantization)
  - v2qqpq (v2qq with pwconv quantization)
  - v2f (fixed index $k$ during retraining time with v2qqpq)
  - v2nb (no $\beta$ with v2qqpq)
'''
versions = [
    'v2', 'v2q', 'v2qq', 'v2qpq', 'v2qqpq', 'v2f', 'v2nb',
]


def config():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='KH Research')
    parser.add_argument('dataset', metavar='DATA', default='cifar10',
                        choices=dataset_names,
                        help='dataset: ' +
                             ' | '.join(dataset_names) +
                             ' (default: cifar10)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: mobilenet)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (defualt: 0.1)',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('--layers', default=16, type=int, metavar='N',
                        help='number of layers in VGG/ResNet/ResNeXt/WideResNet (default: 16)')
    parser.add_argument('--bn', '--batch-norm', dest='bn', action='store_true',
                        help='Use batch norm in VGG?')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='Path of checkpoint for resuming/testing '
                             'or retraining model (Default: none)')
    parser.add_argument('-R', '--resume', dest='resume', action='store_true',
                        help='Resume model?')
    parser.add_argument('-E', '--evaluate', dest='evaluate', action='store_true',
                        help='Test model?')
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='Use cuda?')
    parser.add_argument('-T', '--retrain', dest='retrain', action='store_true',
                        help='Retraining?')
    parser.add_argument('-g', '--gpuids', default=[0], nargs='+',
                        help='GPU IDs for using (Default: 0)')
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')
    # for new methods
    parser.add_argument('-v', '--version', default='', metavar='V', dest='version',
                        choices=versions,
                        help='version: ' +
                             ' | '.join(versions) +
                             ' (find kernel version (default: none))')
    parser.add_argument('-d', '--bind-size', default=2, type=int, metavar='N',
                        dest='bind_size',
                        help='the number of binding channels in convolution '
                             '(subvector size) on version 3 (v3) (default: 2)')
    parser.add_argument('-pwd', '--pw-bind-size', default=8, type=int, metavar='N',
                        dest='pw_bind_size',
                        help='the number of binding channels in pointwise convolution '
                             '(subvector size) (default: 8)')
    parser.add_argument('-N', '--new', dest='new', action='store_true',
                        help='new method?')
    parser.add_argument('-s', '--save-epoch', default=5, type=int, metavar='N',
                        dest='save_epoch',
                        help='number of epochs to save checkpoint and to apply new method')
    parser.add_argument('--nl', '--nuc-loss', dest='nuc_loss', action='store_true',
                        help='nuclear norm loss?')
    parser.add_argument('--nls', '--nl-scale', dest='nls', default=1.0, type=float,
                        help='scale factor of nuc_loss')
    parser.add_argument('--pl', '--pcc-loss', dest='pcc_loss', action='store_true',
                        help='pearson correlation coefficient loss?')
    parser.add_argument('--pls', '--pl-scale', dest='pls', default=1.0, type=float,
                        help='scale factor of pcc_loss')
    # for quantization
    parser.add_argument('-Q', '--quant', dest='quant', action='store_true',
                        help='use quantization?')
    parser.add_argument('--np', action='store_true',
                        help='no v2-like method in pointwise convolutional layer?')
    parser.add_argument('--qb', '--quant-bit', default=8, type=int, metavar='N', dest='quant_bit',
                        help='number of bits for quantization (Default: 8)')
    parser.add_argument('--qba', '--quant_bit_a', default=8, type=int, metavar='N', dest='quant_bit_a',
                        help='number of bits for quantizing alphas (Default: 8)')
    parser.add_argument('--qbb', '--quant_bit_b', default=8, type=int, metavar='N', dest='quant_bit_b',
                        help='number of bits for quantizing betas (Default: 8)')

    cfg = parser.parse_args()
    cfg.gpuids = list(map(int, cfg.gpuids))
    return cfg

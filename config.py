import argparse
import models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

r'''version list
- v1
- v2
  - v2
  - v2q ($\alpha$, $\beta$ quantization and adding epsilon in denom of $\alpha$)
  - v2qq (quantization ver with $\alpha$, $\beta$ quantization and adding epsilon in denom of $\alpha$)
  - v2f (fixed index $k$ during retraining time with v2qq)
  - v2qnb (no $\beta$ with v2q)
  - v2qqnb (no $\beta$ with v2qq)
  - v2.5 (old: v3)
- v3: rotation, flip, shift
'''
versions = [
    'v1',
    'v2', 'v2q', 'v2qq', 'v2f', 'v2nb', 'v2qnb', 'v2qqnb',
    # 'v2.5',
    # 'v3',
]

# sacred setting
MONGO_URI = 'mongodb://mlvc:mlvcdatabase!@mlvc.khu.ac.kr:31912'
MONGO_DB = 'training'


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
    parser.add_argument('--layers', default=16, type=int, metavar='N',
                        help='number of layers in VGG/ResNet/WideResNet (default: 16)')
    parser.add_argument('--bn', '--batch-norm', dest='bn', action='store_true',
                        help='Use batch norm in VGG?')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--dr', '--drop-rate', default=0.3, type=float,
                        metavar='DR', help='dropout rate for WRN (default: 0.3)',
                        dest='drop_rate')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    # parser.add_argument('--nesterov', dest='nesterov', action='store_true',
    #                     help='use nesterov momentum?')
    parser.add_argument('--basetest', dest='basetest', action='store_true',
                        help='baseline test (various weight decay test) (\'_wd\{\}\' added in filename)')
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
                             '(subvector size) on version 2.5 (v2.5) (default: 2)')
    parser.add_argument('-pwd', '--pw-bind-size', default=8, type=int, metavar='N',
                        dest='pw_bind_size',
                        help='the number of binding channels in pointwise convolution '
                             '(subvector size) (default: 8)')
    parser.add_argument('-pws', '--pwkernel-stride', default=1, type=int, metavar='N',
                        dest='pwkernel_stride',
                        help='the number of pwkernel stride size in reference layer '
                             '(default: 1)')
    parser.add_argument('-N', '--new', dest='new', action='store_true',
                        help='new method?')
    parser.add_argument('-refnum', '--ref_layer_num', dest='refnum', default=0, type=int, metavar='N',
                        help='reference layer number (default: 0)')
    parser.add_argument('-eps', '--epsilon', dest='epsilon', default=1e-08, type=float, metavar='EPS',
                        help='epsilon for denominator of alpha in find_kernel (default: 1e-08)')
    parser.add_argument('-s', '--save-epoch', dest='save_epoch', default=1, type=int, metavar='N',
                        help='number of epochs to save checkpoint and to apply new method (default: 1)')
    parser.add_argument('--tvl', '--tv-loss', dest='tv_loss', action='store_true',
                        help='total variation loss?')
    parser.add_argument('--tvls', '--tvl-scale', dest='tvls', default=1e-08, type=float,
                        help='scale factor of tv_loss (default: 1e-08)')
    parser.add_argument('--orthol', '--ortho-loss', dest='ortho_loss', action='store_true',
                        help='orthogonal loss?')
    parser.add_argument('--orthols', '--orthol-scale', dest='orthols', default=1e-05, type=float,
                        help='scale factor of ortho_loss (default: 1e-05)')
    parser.add_argument('--corl', '--cor-loss', dest='cor_loss', action='store_true',
                        help='correlation loss?')
    parser.add_argument('--corls', '--corl-scale', dest='corls', default=1e-05, type=float,
                        help='scale factor of cor_loss (default: 1e-05)')
    parser.add_argument('--orthocorl', '--ortho-cor-loss', dest='ortho_cor_loss', action='store_true',
                        help='orthogonal correlation loss?')
    parser.add_argument('--orthocorls', '--orthocorl-scale', dest='orthocorls', default=1e-05, type=float,
                        help='scale factor of ortho_cor_loss (default: 1e-05)')
    parser.add_argument('--groupcorl', '--groupcor-loss', dest='groupcor_loss', action='store_true',
                        help='group correlation loss?')
    parser.add_argument('--groupcorls', '--groupcorl-scale', dest='groupcorls', default=1e-05, type=float,
                        help='scale factor of groupcor_loss (default: 1e-05)')
    parser.add_argument('--groupcorn', '--groupcor-num', dest='groupcor_num', default=1, type=int,
                        help='the number of kernel sets for groups of groupcor_loss (if this number is n, group size is n*3) '
                             '(available number: 1,2,4,8,16) (default: 1)')
    parser.add_argument('-warm', '--warmup-epoch', dest='warmup_epoch', default=0, type=int, metavar='N',
                        help='number of warmup epochs for applying the V2 method (default: 0)')
    parser.add_argument('--w-anal', '--weight-analysis', dest='w_anal', action='store_true',
                        help='weight analysis in find_similar_kernel.py')
    # for quantization
    parser.add_argument('-Q', '--quant', dest='quant', action='store_true',
                        help='use quantization?')
    parser.add_argument('--qb', '--quant-bit', default=8, type=int, metavar='N', dest='quant_bit',
                        help='number of bits for quantization (default: 8)')
    parser.add_argument('--qba', '--quant_bit_a', default=8, type=int, metavar='N', dest='quant_bit_a',
                        help='number of bits for quantizing alphas (default: 8)')
    parser.add_argument('--qbb', '--quant_bit_b', default=8, type=int, metavar='N', dest='quant_bit_b',
                        help='number of bits for quantizing betas (default: 8)')
    # for analysis
    parser.add_argument('--chk-save', dest='chk_save', action='store_true',
                        help='save indices and kernel weight for check index variation at fine-tuning')
    parser.add_argument('--chk-num', default=0, type=int, metavar='N', dest='chk_num',
                        help='number of channel number for check index variation at fine-tuning')

    cfg = parser.parse_args()
    cfg.gpuids = list(map(int, cfg.gpuids))
    return cfg

import time
import pathlib
import argparse
from os.path import isfile
from tqdm import tqdm

import torch
import numpy as np

import models
from utils import load_model
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def config():
    parser = argparse.ArgumentParser(description='Linear quantization')
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
                        help='number of layers in VGG/ResNet/ResNeXt/WideResNet (default: 16)')
    parser.add_argument('--bn', '--batch-norm', dest='bn', action='store_true',
                        help='Use batch norm in VGG?')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--groups', default=2, type=int, metavar='N',
                        help='number of groups for ShuffleNet (default: 2)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='Path of checkpoint (Default: none)')
    parser.add_argument('--qb', '--quant_bit', default=8, type=int, metavar='N', dest='quant_bit',
                        help='number of bits for quantization (Default: 8)')
    parser.add_argument('--pq', action='store_true',
                        help='pointwise convolution quantization in baseline?')
    parser.add_argument('-i', '--ifl', dest='ifl', action='store_true',
                        help='include first layer?')

    cfg = parser.parse_args()
    return cfg


def main():
    global opt, dir_path, hasDiffLayersArchs
    opt = config()

    arch_name = opt.arch
    hasDiffLayersArchs = ['vgg', 'resnet', 'resnext', 'wideresnet']
    if opt.arch in hasDiffLayersArchs:
        arch_name += str(opt.layers)

    print('\n=> creating model \'{}\''.format(arch_name))
    model = models.__dict__[opt.arch](data=opt.dataset,
                                      num_layers=opt.layers, num_groups=opt.groups,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)

    # checkpoint file
    ckpt_dir = pathlib.Path('checkpoint')
    dir_path = ckpt_dir / arch_name / opt.dataset
    ckpt_file = dir_path / opt.ckpt

    if isfile(ckpt_file):
        print('==> Loading Checkpoint \'{}\''.format(opt.ckpt))
        checkpoint = load_model(model, ckpt_file, None, use_cuda=False)

        print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                    opt.ckpt, checkpoint['epoch']))

        print('==> Quantize weights at Checkpoint \'{}\''.format(opt.ckpt))
        new_ckpt_name = save_quantized_model(model, checkpoint, opt.quant_bit)
        print('===> Save new Checkpoint \'{}\''.format(new_ckpt_name))
        return
    else:
        print('==> no Checkpoint found at \'{}\''.format(
                    opt.ckpt))
        return


def save_quantized_model(model, ckpt, num_bits=8):
    """save quantized model"""
    if opt.arch in hasDiffLayersArchs:
        w_conv = model.get_weights_conv(use_cuda=False)
    else:
        w_conv = model.get_weights_dwconv(use_cuda=False)
        if opt.pq:
            w_pwconv = model.get_weights_pwconv(use_cuda=False)

    num_layer = len(w_conv)

    qmin = -2.**(num_bits - 1.)
    qmax = 2.**(num_bits - 1.) - 1.

    if opt.ifl:
        start_layer = 0
    else:
        start_layer = 1

    for i in tqdm(range(start_layer, num_layer), ncols=80, unit='layer'):
        min_val = np.amin(w_conv[i])
        max_val = np.amax(w_conv[i])
        scale = (max_val - min_val) / (qmax - qmin)
        w_conv[i] = np.around(np.clip(w_conv[i] / scale, qmin, qmax))
        w_conv[i] = scale * w_conv[i]
    
    if opt.pq:
        num_layer = len(w_pwconv)

        qmin = -2.**(num_bits - 1.)
        qmax = 2.**(num_bits - 1.) - 1.

        if opt.ifl:
            start_layer = 0
        else:
            start_layer = 1

        for i in tqdm(range(start_layer, num_layer), ncols=80, unit='layer'):
            min_val = np.amin(w_pwconv[i])
            max_val = np.amax(w_pwconv[i])
            scale = (max_val - min_val) / (qmax - qmin)
            w_pwconv[i] = np.around(np.clip(w_pwconv[i] / scale, qmin, qmax))
            w_pwconv[i] = scale * w_pwconv[i]

    if opt.arch in hasDiffLayersArchs:
        model.set_weights_conv(w_conv, use_cuda=False)
    else:
        model.set_weights_dwconv(w_conv, use_cuda=False)
        if opt.pq:
            w_pwconv = model.set_weights_pwconv(use_cuda=False)

    ckpt['model'] = model.state_dict()

    new_model_filename = '{}_q{}'.format(opt.ckpt[:-4], opt.quant_bit)
    if opt.pq:
        new_model_filename += '_pq'
    if opt.ifl:
        new_model_filename += '_ifl'
    new_model_filename += '.pth'
    model_file = dir_path / new_model_filename

    torch.save(ckpt, model_file)
    return new_model_filename


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('====> total time: {:.2f}s'.format(elapsed_time))

import time
import pathlib
import argparse
from os.path import isfile
from tqdm import tqdm

import torch
import numpy as np

import models
from config import config
from utils import hasDiffLayersArchs, hasPWConvArchs, load_model, get_kernel, get_pwkernel, set_kernel, set_pwkernel


def main():
    global opt, arch_name, dir_path
    opt = config()

    # quantization don't need cuda
    if opt.cuda:
        print('==> just apply linear quantization don\'t need cuda option. exit..\n')
        exit()

    # set model name
    arch_name = opt.arch
    if opt.arch in hasDiffLayersArchs:
        arch_name += str(opt.layers)

    print('\n=> creating model \'{}\''.format(arch_name))
    model = models.__dict__[opt.arch](data=opt.dataset,
                                      num_layers=opt.layers, num_groups=opt.groups,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)

    if model is None:
        print('==> unavailable model parameters!! exit...\n')
        exit()

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
        exit()


def save_quantized_model(model, ckpt, num_bits=8):
    """save quantized model"""
    quantize(model, opt, num_bits=num_bits)
    if arch_name in hasPWConvArchs:
        quantize_pw(model, opt, num_bits=num_bits)

    ckpt['model'] = model.state_dict()

    new_model_filename = '{}_q{}'.format(opt.ckpt[:-4], opt.quant_bit)
    new_model_filename += '.pth'
    model_file = dir_path / new_model_filename

    torch.save(ckpt, model_file)
    return new_model_filename


def quantize(model, opt, num_bits=8):
    r"""quantize weights of convolution kernels
    """
    w_kernel = get_kernel(model, opt)
    num_layer = len(w_kernel)

    qmin = -2.**(num_bits - 1.)
    qmax = 2.**(num_bits - 1.) - 1.

    for i in tqdm(range(num_layer), ncols=80, unit='layer'):
        min_val = np.amin(w_kernel[i])
        max_val = np.amax(w_kernel[i])
        scale = (max_val - min_val) / (qmax - qmin)
        w_kernel[i] = np.around(np.clip(w_kernel[i] / scale, qmin, qmax))
        w_kernel[i] = scale * w_kernel[i]
    
    set_kernel(w_kernel, model, opt)


def quantize_pw(model, opt, num_bits=8):
    r"""quantize weights of pointwise covolution kernels
    """
    w_kernel = get_pwkernel(model, opt)
    num_layer = len(w_kernel)

    qmin = -2.**(num_bits - 1.)
    qmax = 2.**(num_bits -1.) -1.

    for i in tqdm(range(num_layer), ncols=80, unit='layer'):
        min_val = np.amin(w_kernel[i])
        max_val = np.amax(w_kernel[i])
        scale = (max_val - min_val) / (qmax - qmin)
        w_kernel[i] = np.around(np.clip(w_kernel[i] / scale, qmin, qmax))
        w_kernel[i] = scale * w_kernel[i]
    
    set_pwkernel(w_kernel, model, opt)


def quantize_ab(indices, num_bits_a=8, num_bits_b=8):
    r"""quantize $\alpha$ and $\beta$
    """
    qmin_a = -2.**(num_bits_a - 1.)
    qmax_a = 2.**(num_bits_a - 1.) - 1.
    qmin_b = -2.**(num_bits_b - 1.)
    qmax_b = 2.**(num_bits_b - 1.) - 1.

    for i in tqdm(range(len(indices)), ncols=80, unit='layer'):
        k = []
        alphas = []
        betas = []
        for j in range(len(indices[i])):
            _k, _alpha, _beta = indices[i][j]
            k.append(_k)
            alphas.append(_alpha)
            betas.append(_beta)
        min_val_a = np.amin(alphas)
        max_val_a = np.amax(alphas)
        min_val_b = np.amin(betas)
        max_val_b = np.amax(betas)
        scale_a = (max_val_a - min_val_a) / (qmax_a - qmin_a)
        scale_b = (max_val_b - min_val_b) / (qmax_b - qmin_b)
        alphas = np.around(np.clip(alphas / scale_a, qmin_a, qmax_a))
        betas = np.around(np.clip(betas / scale_b, qmin_b, qmax_b))
        alphas = scale_a * alphas
        betas = scale_b * betas
        for j in range(len(indices[i])):
            indices[i][j] = k[j], alphas[j], betas[j]


def quantize_alpha(indices, num_bits_a=8):
    r"""quantize $\alpha$
    """
    qmin_a = -2.**(num_bits_a - 1.)
    qmax_a = 2.**(num_bits_a - 1.) - 1.

    for i in tqdm(range(len(indices)), ncols=80, unit='layer'):
        k = []
        alphas = []
        betas = []
        for j in range(len(indices[i])):
            _k, _alpha, _beta = indices[i][j]
            k.append(_k)
            alphas.append(_alpha)
            betas.append(_beta)
        min_val_a = np.amin(alphas)
        max_val_a = np.amax(alphas)
        scale_a = (max_val_a - min_val_a) / (qmax_a - qmin_a)
        alphas = np.around(np.clip(alphas / scale_a, qmin_a, qmax_a))
        alphas = scale_a * alphas
        for j in range(len(indices[i])):
            indices[i][j] = k[j], alphas[j], betas[j]


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('====> total time: {:.2f}s'.format(elapsed_time))

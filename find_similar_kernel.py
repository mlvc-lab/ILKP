'''
TODO:
저장을 어떻게 할 것인가
원래 코드에서 model.state_dict()만 저장하고 불러들이는데 사용됨.
conv weight만 따로 빼서 저장해보기
새로운 방법은 index 따로 weight 따로 저장.
main.py에서 불러들이는 거 구현
'''

import time
import pathlib
import argparse
from os.path import isfile
from tqdm import tqdm

import math
import torch
import numpy as np
from copy import deepcopy

import models
from config import config
from utils import hasDiffLayersArchs, hasPWConvArchs, load_model, set_arch_name, get_kernel
from quantize import quantize, quantize_ab


def main():
    global opt, arch_name, dir_path
    opt = config()

    # finding similar kernels doesn't need cuda
    if opt.cuda:
        print('==> finding similar kernels doesn\'t need cuda option. exit..\n')
        exit()

    # set model name
    arch_name = set_arch_name(opt)

    print('\n=> creating model \'{}\''.format(arch_name))
    model = models.__dict__[opt.arch](data=opt.dataset, num_layers=opt.layers,
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
        checkpoint = load_model(model, ckpt_file,
                                main_gpu=None, use_cuda=False)
        print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
            opt.ckpt, checkpoint['epoch']))

        if opt.w_anal:
            if not opt.version == 'v2':
                print('analysis can only be used with ver2...')
                exit()
            weight_analysis(model, checkpoint)
            return
        if opt.version in ['v2q', 'v2qq', 'v2f', 'v2nb', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
            print('==> {}bit Quantization...'.format(opt.quant_bit))
            quantize(model, opt, opt.quant_bit)
            if arch_name in hasPWConvArchs and not opt.np:
                quantize(model, opt, opt.quant_bit, is_pw=True)
        print('==> Find the most similar kernel in reference layers ' +
              'from filters at Checkpoint \'{}\''.format(opt.ckpt))
        indices = find_kernel(model, opt)
        if arch_name in hasPWConvArchs and not opt.np:
            indices_pw = find_kernel_pw(model, opt)
            indices = (indices, indices_pw)
        new_ckpt_name = save_model(checkpoint, indices)
        print('===> Save new Checkpoint \'{}\''.format(new_ckpt_name))
        return
    else:
        print('==> no Checkpoint found at \'{}\''.format(
            opt.ckpt))
        exit()


def find_kernel(model, opt):
    r"""Find the most similar kernel

    Return:
        idx_all (list): indices of similar kernels with $\alpha$ and $\beta$.
    """
    w_kernel = get_kernel(model, opt)
    num_layer = len(w_kernel)

    ref_layer_num = 0
    idx_all = []

    ref_layer = torch.Tensor(w_kernel[ref_layer_num])
    # change kernels to dw-kernel
    if opt.arch in hasDiffLayersArchs:
        ref_layer = ref_layer.view(-1, 9)
    else:
        ref_layer = ref_layer.view(len(w_kernel[ref_layer_num]), -1)
    ref_length = ref_layer.size()[0]
    ref_mean = ref_layer.mean(dim=1, keepdim=True)
    ref_norm = ref_layer - ref_mean
    denom = (ref_norm * ref_norm).sum(dim=1)

    epsilon = opt.epsilon # epsilon for non-zero denom (default: 1e-08)
    if opt.version == 'v2qq-epsv1': # add epsilon to every denom
        denom += epsilon
    elif opt.version == 'v2qq-epsv2': # if denom is 0, set denom to epsilon
        denom[denom.eq(0.0)] = epsilon
    denom = denom.view(-1, ref_length)

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        idx = []
        cur_weight = torch.Tensor(w_kernel[i])
        # change kernels to dw-kernel
        if opt.arch in hasDiffLayersArchs:
            cur_weight = cur_weight.view(-1, 9)
        else:
            cur_weight = cur_weight.view(len(w_kernel[i]), -1)
        cur_length = cur_weight.size()[0]
        cur_mean = cur_weight.mean(dim=1, keepdim=True)
        cur_norm = cur_weight - cur_mean

        for j in range(cur_length):
            numer = torch.matmul(cur_norm[j], ref_norm.T)
            alphas = deepcopy(numer / denom)
            del numer

            if opt.version == 'v2qq-epsv3': # if alpha is nan, set alpha to 1.0
                alphas[alphas.ne(alphas)] = 1.0

            betas = cur_mean[j][0] - alphas * ref_mean.view(-1, ref_length)
            residual_mat = (ref_layer * alphas.view(ref_length, -1) + betas.view(ref_length, -1)) -\
                cur_weight[j].expand_as(ref_layer)
            residual_mat = residual_mat.abs().sum(dim=1)
            k = deepcopy(residual_mat.argmin().item())
            alpha = deepcopy(alphas[0][k].item())
            beta = deepcopy(betas[0][k].item())
            ref_idx = (k, alpha, beta)
            idx.append(ref_idx)
            del alphas, betas, residual_mat

        del cur_weight, cur_norm, cur_mean, cur_length
        idx_all.append(idx)

    del ref_layer, ref_mean, ref_norm, denom

    return idx_all


def find_kernel_pw(model, opt):
    r"""Find the most similar kernel in pointwise convolutional layers using `cuda`

    Return:
        idx_all (list): indices of similar kernels with $\alpha$ and $\beta$.
    """
    w_kernel = get_kernel(model, opt, is_pw=True)
    num_layer = len(w_kernel)
    pwd = opt.pw_bind_size
    pws = opt.pwkernel_stride
    idx_all = []

    ref_layer_num = 0
    ref_layer = torch.Tensor(w_kernel[ref_layer_num]).cuda()
    # ref_layer = torch.Tensor(w_kernel[ref_layer_num])
    ref_layer = ref_layer.view(ref_layer.size(0), ref_layer.size(1))
    ref_layer_slices = None
    num_slices = (ref_layer.size(1) - pwd) // pws + 1
    for i in range(0, ref_layer.size(1) - pwd + 1, pws):
        if ref_layer_slices == None:
            ref_layer_slices = ref_layer[:, i:i+pwd]
        else:
            ref_layer_slices = torch.cat((ref_layer_slices, ref_layer[:, i:i+pwd]), dim=1)
    if ((ref_layer.size(1) - pwd) % pws) != 0:
        ref_layer_slices = torch.cat((ref_layer_slices, ref_layer[:, -pwd:]), dim=1)
        num_slices += 1
    ref_layer_slices = ref_layer_slices.view(ref_layer.size(0)*num_slices, pwd)
    ref_length = ref_layer_slices.size(0)
    ref_mean = ref_layer_slices.mean(dim=1, keepdim=True)
    ref_norm = ref_layer_slices - ref_mean
    _denom = (ref_norm * ref_norm).sum(dim=1)

    epsilon = opt.epsilon # epsilon for non-zero denom (default: 1e-08)
    if opt.version == 'v2qq-epsv1': # add epsilon to every denom
        _denom += epsilon
    elif opt.version == 'v2qq-epsv2': # if denom is 0, set denom to epsilon
        _denom[_denom.eq(0.0)] = epsilon

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        idx = []
        cur_layer = torch.Tensor(w_kernel[i]).cuda()
        # cur_layer = torch.Tensor(w_kernel[i])
        cur_layer = cur_layer.view(cur_layer.size(0), -1)
        cur_layer_length = cur_layer.size(0)
        for j in range(cur_layer_length):
            cur_weight = cur_layer[j].view(-1, pwd)
            cur_length = cur_weight.size(0)
            cur_mean = cur_weight.mean(dim=1, keepdim=True)
            cur_norm = cur_weight - cur_mean

            numer = torch.matmul(cur_norm, ref_norm.T)
            denom = deepcopy(_denom.expand_as(numer))
            alphas = deepcopy(numer / denom)
            del numer, denom

            if opt.version == 'v2qq-epsv3': # if alpha is nan, set alpha to 1.0
                alphas[alphas.ne(alphas)] = 1.0

            betas = cur_mean - alphas * ref_mean.view(-1, ref_length).expand_as(alphas)
            for idx_cur_slice in range(cur_length):
                cur_alphas = alphas[idx_cur_slice].view(ref_length, -1)
                cur_betas = betas[idx_cur_slice].view(ref_length, -1)
                residual_mat = (ref_layer_slices * cur_alphas + cur_betas) -\
                    cur_weight[idx_cur_slice].expand_as(ref_layer_slices)
                residual_mat = residual_mat.abs().sum(dim=1)
                k = deepcopy(residual_mat.argmin().cpu().item())
                alpha = deepcopy(alphas[idx_cur_slice][k].cpu().item())
                beta = deepcopy(betas[idx_cur_slice][k].cpu().item())
                # k = deepcopy(residual_mat.argmin().item())
                # alpha = deepcopy(alphas[idx_cur_slice][k].item())
                # beta = deepcopy(betas[idx_cur_slice][k].item())
                ref_idx = (k, alpha, beta)
                idx.append(ref_idx)

            del alphas, betas, cur_alphas, cur_weight, cur_betas, cur_norm, cur_mean, cur_length
            torch.cuda.empty_cache()

        del cur_layer
        torch.cuda.empty_cache()
        idx_all.append(idx)

    del ref_layer, ref_mean, ref_norm, _denom

    return idx_all


def save_model(ckpt, indices_all):
    r"""Save new model
    """
    if arch_name in hasPWConvArchs and not opt.np:
        indices, indices_pw = indices_all
    else:
        indices = indices_all

    if opt.version in ['v2qq', 'v2f', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
        quantize_ab(indices, num_bits_a=opt.quant_bit_a,
                    num_bits_b=opt.quant_bit_b)
    elif opt.version == 'v2nb':
        quantize_ab(indices, num_bits_a=opt.quant_bit_a)
    if arch_name in hasPWConvArchs and not opt.np:
        if opt.version in ['v2qq', 'v2f', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
            quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a,
                        num_bits_b=opt.quant_bit_b)
        elif opt.version == 'v2nb':
            quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a)
        indices = (indices, indices_pw)

    ckpt['idx'] = indices
    ckpt['version'] = opt.version
    new_model_filename = '{}_{}'.format(opt.ckpt[:-4], opt.version)
    if opt.np:
        file_name += '_np'
    if opt.version in ['v2q', 'v2qq', 'v2f', 'v2nb', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
        new_model_filename += '_q{}'.format(opt.quant_bit)
        if opt.version in ['v2qq', 'v2f', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
            new_model_filename += '{}{}'.format(
                opt.quant_bit_a, opt.quant_bit_b)
            if opt.version in ['v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
                new_model_filename += '_eps{}'.format(
                    opt.epsilon)
        elif opt.version == 'v2nb':
            new_model_filename += '{}'.format(
                opt.quant_bit_a)
    # elif opt.version in ['v3', 'v3a']:
    #     new_model_filename += '_d{}'.format(opt.bind_size)
    new_model_filename += '.pth'
    model_file = dir_path / new_model_filename
    torch.save(ckpt, model_file)

    return new_model_filename


def weight_analysis(model, ckpt):
    r"""Analysis of dwkernel weights
    """
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils import hasDWConvArchs
    # make directory for saving plots
    dir_weights = pathlib.Path('plot_weights')
    dir_weights.mkdir(parents=True, exist_ok=True)

    indices = find_kernel(model, opt)
    w_kernel = get_kernel(model, opt)

    ref_layer_num = 0
    layerNumList = list(range(1, len(w_kernel)))
    randLayerIdx = random.sample(layerNumList, 5)
    for i in tqdm(randLayerIdx, ncols=80, unit='layer'):
        kernelNumList = list(range(len(w_kernel[i])))
        randKernelIdx = random.sample(kernelNumList, 3)
        for j in randKernelIdx:
            dwkernelNumList = list(range(len(w_kernel[i][j])))
            randdwKernelIdx = random.sample(dwkernelNumList, 1)
            for k in randdwKernelIdx:
                ref_idx = indices[i-1][j*len(w_kernel[i][j])+k][0]
                v = ref_idx // len(w_kernel[ref_layer_num][0])
                w = ref_idx % len(w_kernel[ref_layer_num][0])
                weights_cur = np.reshape(w_kernel[i][j][k], -1)
                weights_ref = np.reshape(w_kernel[ref_layer_num][v][w], -1)
                plt.figure(figsize=(8,6), dpi=300)
                plt.title('{}-{}'.format(arch_name, opt.dataset))
                if opt.arch in hasDWConvArchs:
                    plt.xlabel(r'current kernel $K_{{ {},{} }}$'.format(i,j))
                    plt.ylabel(r'reference kernel $K_{{ {},{} }}$'.format(ref_layer_num,ref_idx))
                    plot_name = '{}_{}_weight_{}_{}.png'.format(arch_name, opt.dataset, i, j)
                else:
                    plt.xlabel(r'current kernel $K_{{ {},{},{} }}$'.format(i,j,k))
                    plt.ylabel(r'reference kernel $K_{{ {},{},{} }}$'.format(ref_layer_num,v,w))
                    plot_name = '{}_{}_weight_{}_{}_{}.png'.format(arch_name, opt.dataset, i, j, k)
                plt.scatter(weights_cur, weights_ref)
                plt.savefig(dir_weights / plot_name, bbox_inches='tight')


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("====> total time: {:.2f}s".format(elapsed_time))

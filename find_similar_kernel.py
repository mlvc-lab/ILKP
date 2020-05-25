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
from utils import hasDiffLayersArchs, hasPWConvArchs, load_model, get_kernel, get_pwkernel, set_kernel, set_pwkernel
from quantize import quantize, quantize_pw, quantize_ab, quantize_alpha


def main():
    global opt, arch_name, dir_path
    opt = config()

    # finding similar kernels doesn't need cuda
    if opt.cuda:
        print('==> finding similar kernels doesn\'t need cuda option. exit..\n')
        exit()

    # set model name
    arch_name = opt.arch
    if opt.arch in hasDiffLayersArchs:
        arch_name += str(opt.layers)

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

        # if opt.analysis:
        #     if not opt.version == 'v2':
        #         print('analysis can only be used with ver2...')
        #         exit()
        #     weight_analysis(model, checkpoint)
        #     return
        if opt.version in ['v2q', 'v2qq', 'v2f', 'v2nb']:
            print('==> {}bit Quantization...'.format(opt.quant_bit))
            quantize(model, opt, opt.quant_bit)
            if arch_name in hasPWConvArchs and not opt.np:
                quantize_pw(model, opt, opt.quant_bit)
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
    r"""find the most similar kernel
    """
    w_kernel = get_kernel(model, opt)
    num_layer = len(w_kernel)

    ref_layer_num = 0
    idx_all = []

    if opt.arch == 'vgg':
        ref_layer = torch.Tensor(w_kernel[ref_layer_num]).cuda()
    else:
        ref_layer = torch.Tensor(w_kernel[ref_layer_num])
    # change kernels to dw-kernel
    if opt.arch in hasDiffLayersArchs:
        ref_layer = ref_layer.view(-1, 9)
    else:
        ref_layer = ref_layer.view(len(w_kernel[ref_layer_num]), -1)
    ref_length = ref_layer.size()[0]
    ref_mean = ref_layer.mean(dim=1, keepdim=True)
    ref_norm = ref_layer - ref_mean
    ref_norm_sq = (ref_norm * ref_norm).sum(dim=1)

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        idx = []
        if opt.arch == 'vgg':
            cur_weight = torch.Tensor(w_kernel[i]).cuda()
        else:
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
            denom = ref_norm_sq.view(-1, ref_length)
            alphas = deepcopy(numer / denom)
            del numer, denom
            betas = cur_mean[j][0] - alphas * ref_mean.view(-1, ref_length)
            residual_mat = (ref_layer * alphas.view(ref_length, -1) + betas.view(ref_length, -1)) -\
                cur_weight[j].expand_as(ref_layer)
            residual_mat = residual_mat.abs().sum(dim=1)
            if opt.arch == 'vgg':
                k = deepcopy(residual_mat.argmin().cpu().item())
                alpha = deepcopy(alphas[0][k].cpu().item())
                beta = deepcopy(betas[0][k].cpu().item())
            else:
                k = deepcopy(residual_mat.argmin().item())
                alpha = deepcopy(alphas[0][k].item())
                beta = deepcopy(betas[0][k].item())
            ref_idx = (k, alpha, beta)
            idx.append(ref_idx)
            del alphas, betas, residual_mat
            if opt.arch == 'vgg':
                torch.cuda.empty_cache()
        del cur_weight, cur_norm, cur_mean, cur_length
        if opt.arch == 'vgg':
            torch.cuda.empty_cache()
        idx_all.append(idx)
    del ref_layer, ref_mean, ref_norm, ref_norm_sq

    return idx_all


def find_kernel_pw(model, opt):
    r"""find the most similar kernel in pointwise convolutional layers
    """
    w_kernel = get_pwkernel(model, opt)
    num_layer = len(w_kernel)
    d = opt.pw_bind_size
    idx_all = []

    ref_layer_num = 0
    ref_layer = torch.Tensor(w_kernel[ref_layer_num]).cuda()
    # ref_layer = torch.Tensor(w_kernel[ref_layer_num])
    ref_layer = ref_layer.view(ref_layer.size(0), ref_layer.size(1))
    ref_layer_slices = None
    num_slices_per_kernel = ref_layer.size(1) - d + 1
    for i in range(num_slices_per_kernel):
        if ref_layer_slices == None:
            ref_layer_slices = ref_layer[:,i:i+d]
        else:
            ref_layer_slices = torch.cat((ref_layer_slices, ref_layer[:,i:i+d]), dim=1)
    ref_layer_slices = ref_layer_slices.view(ref_layer.size(0)*num_slices_per_kernel, d)
    ref_length = ref_layer_slices.size(0)
    ref_mean = ref_layer_slices.mean(dim=1, keepdim=True)
    ref_norm = ref_layer_slices - ref_mean
    ref_norm_sq = (ref_norm * ref_norm).sum(dim=1)

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        idx = []
        cur_layer = torch.Tensor(w_kernel[i]).cuda()
        # cur_layer = torch.Tensor(w_kernel[i])
        cur_layer = cur_layer.view(cur_layer.size(0), -1)
        cur_layer_length = cur_layer.size(0)
        for j in range(cur_layer_length):
            cur_weight = cur_layer[j].view(-1, d)
            cur_length = cur_weight.size(0)
            cur_mean = cur_weight.mean(dim=1, keepdim=True)
            cur_norm = cur_weight - cur_mean

            numer = torch.matmul(cur_norm, ref_norm.T)
            denom = ref_norm_sq.expand_as(numer)
            alphas = deepcopy(numer / denom)
            del numer, denom
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
    del ref_layer, ref_mean, ref_norm, ref_norm_sq

    return idx_all


def save_model(ckpt, indices_all):
    r"""save new model
    """
    if arch_name in hasPWConvArchs and not opt.np:
        indices, indices_pw = indices_all
    else:
        indices = indices_all

    if opt.version in ['v2qq', 'v2f']:
        quantize_ab(indices, num_bits_a=opt.quant_bit_a, num_bits_b=opt.quant_bit_b)
    elif opt.version == 'v2nb':
        quantize_alpha(indices, num_bits_a=opt.quant_bit_a)
    if arch_name in hasPWConvArchs and not opt.np:
        if opt.version in ['v2qq', 'v2f']:
            quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a, num_bits_b=opt.quant_bit_b)
        elif opt.version == 'v2nb':
            quantize_alpha(indices_pw, num_bits_a=opt.quant_bit_a)
        indices = (indices, indices_pw)

    ckpt['idx'] = indices
    ckpt['version'] = opt.version
    new_model_filename = '{}_{}'.format(opt.ckpt[:-4], opt.version)
    if opt.np:
        file_name += '_np'
    if opt.version in ['v2q', 'v2qq', 'v2f', 'v2nb']:
        new_model_filename += '_q{}'.format(opt.quant_bit)
        if opt.version in ['v2qq', 'v2f']:
            new_model_filename += '{}{}'.format(
                opt.quant_bit_a, opt.quant_bit_b)
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
    r"""weight analysis
    """
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dir_weights = pathlib.Path('weights')
    dir_weights.mkdir(parents=True, exist_ok=True)

    w_kernel = model.get_weights_dwconv(use_cuda=False)
    ref_layer_num = 0
    idx_all = []

    # ref_layer = torch.Tensor(w_kernel[ref_layer_num]).cuda()
    ref_layer = torch.Tensor(w_kernel[ref_layer_num])
    if opt.arch in hasDiffLayersArchs:
        ref_layer = ref_layer.view(-1, 9)
    else:
        ref_layer = ref_layer.view(len(w_kernel[ref_layer_num]), -1)
    ref_length = ref_layer.size()[0]
    ref_mean = ref_layer.mean(dim=1).view(ref_length, -1)
    ref_norm = ref_layer - ref_mean
    ref_norm_sq = (ref_norm * ref_norm).sum(dim=1)

    for i in range(1, num_layer):
        idx = []
        # cur_weight = torch.Tensor(w_kernel[i]).cuda()
        cur_weight = torch.Tensor(w_kernel[i])
        print(len(cur_weight))
        if opt.arch in hasDiffLayersArchs:
            cur_weight = cur_weight.view(-1, 9)
        else:
            cur_weight = cur_weight.view(len(w_kernel[i]), -1)
        cur_length = cur_weight.size()[0]
        cur_mean = cur_weight.mean(dim=1).view(cur_length, -1)
        cur_norm = cur_weight - cur_mean

        for j in tqdm(range(cur_length), ncols=80, unit='filter'):
            numer = torch.matmul(cur_norm[j], ref_norm.T)
            denom = ref_norm_sq.view(-1, ref_length)
            alphas = numer / denom
            del numer, denom
            betas = cur_mean[j][0] - alphas * ref_mean.view(-1, ref_length)
            residual_mat = (ref_layer * alphas.view(ref_length, -1) + betas.view(ref_length, -1)) -\
                cur_weight[j].expand_as(ref_layer)
            residual_mat = residual_mat.abs().sum(dim=1)
            # k = deepcopy(residual_mat.argmin().cpu().item())
            # alpha = deepcopy(alphas[0][k].cpu().item())
            # beta = deepcopy(betas[0][k].cpu().item())
            k = deepcopy(residual_mat.argmin().item())
            alpha = deepcopy(alphas[0][k].item())
            beta = deepcopy(betas[0][k].item())
            ref_idx = (k, alpha, beta)
            idx.append(ref_idx)
            del alphas, betas, residual_mat
            # torch.cuda.empty_cache()
        del cur_weight, cur_norm, cur_mean
        # torch.cuda.empty_cache()
        idx_all.append(idx)

    layerNumList = list(range(1, len(w_kernel)))
    randLayerIdx = random.sample(layerNumList,3)
    for i in randLayerIdx:
        kernelNumList = list(range(len(w_kernel[i])))
        randKernelIdx = random.sample(kernelNumList,3)
        for j in randKernelIdx:
            k = idx_all[i-1][j]
            weights_cur = []
            weights_ref = []
            for u in range(3):
                for v in range(3):
                    weights_cur.append(w_kernel[i][j][0][u][v])
                    weights_ref.append(w_kernel[ref_layer_num][k][0][u][v])
            plt.figure(figsize=(8,6), dpi=300)
            plt.title('weights')
            plt.xlabel('current kernel K_{},{}'.format(i,j))
            plt.ylabel('reference kernel K_{},{}'.format(0,k))
            plt.scatter(weights_cur, weights_ref)
            plt.savefig(dir_weights/'{}_{}_weight_{}_{}.png'.format(opt.arch, opt.dataset, i, j),
                        bbox_inches='tight')


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("====> total time: {:.2f}s".format(elapsed_time))

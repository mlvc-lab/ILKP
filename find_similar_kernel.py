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
        if opt.version in ['v2qq', 'v2f', 'v2qqnb']:
            print('==> {}bit Quantization...'.format(opt.quant_bit))
            quantize(model, opt, opt.quant_bit)
            if arch_name in hasPWConvArchs:
                quantize(model, opt, opt.quant_bit, is_pw=True)
        print('==> Find the most similar kernel in reference layers ' +
              'from filters at Checkpoint \'{}\''.format(opt.ckpt))
        indices = find_kernel(model, opt)
        if arch_name in hasPWConvArchs:
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

    ref_layer_num = opt.refnum
    idx_all = []

    ref_layer = torch.Tensor(w_kernel[ref_layer_num])
    if opt.ustv1 == 'sigmoid':
        ref_layer = torch.sigmoid(ref_layer)
    elif opt.ustv1 == 'tanh':
        ref_layer = torch.tanh(ref_layer)

    # change kernels to dw-kernel
    if opt.arch in hasDiffLayersArchs:
        ref_layer = ref_layer.view(-1, 9)
    else:
        ref_layer = ref_layer.view(len(w_kernel[ref_layer_num]), -1)

    ref_length = ref_layer.size()[0]

    ref_mean = ref_layer.mean(dim=1, keepdim=True)
    ref_norm = ref_layer - ref_mean
    ref_norm_sq = (ref_norm * ref_norm).sum(dim=1)
    ref_norm_sq_rt = torch.sqrt(ref_norm_sq)

    # add epsilon if denom is zero
    if opt.version in ['v2q', 'v2qq', 'v2f']:
        alpha_denom = torch.clamp(ref_norm_sq, min=opt.epsilon) # epsilon for non-zero denom (default: 1e-08)
    elif opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
        alpha_denom = (ref_layer * ref_layer).sum(dim=1)
    elif opt.version == 'v2':
        alpha_denom = ref_norm_sq
    
    if opt.version.find('v2') != -1:
        alpha_denom = alpha_denom.view(-1, ref_length)

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        idx = []
        cur_weight = torch.Tensor(w_kernel[i])
        if opt.ustv1 == 'sigmoid':
            cur_weight = torch.sigmoid(cur_weight)
        elif opt.ustv1 == 'tanh':
            cur_weight = torch.tanh(cur_weight)

        # change kernels to dw-kernel
        if opt.arch in hasDiffLayersArchs:
            cur_weight = cur_weight.view(-1, 9)
        else:
            cur_weight = cur_weight.view(len(w_kernel[i]), -1)

        cur_length = cur_weight.size()[0]

        cur_mean = cur_weight.mean(dim=1, keepdim=True)
        cur_norm = cur_weight - cur_mean
        cur_norm_sq_rt = torch.sqrt((cur_norm * cur_norm).sum(dim=1))

        for j in range(cur_length):
            numer = torch.matmul(cur_norm[j], ref_norm.T)
            denom = ref_norm_sq_rt * cur_norm_sq_rt[j]
            pcc = numer / denom

            pcc[pcc.ne(pcc)] = 0.0 # if pcc is nan, set pcc to 0.0
            abs_pcc = torch.abs(pcc)
            k = deepcopy(abs_pcc.argmax().item())

            if opt.version in ['v2', 'v2q', 'v2qq', 'v2f']:
                alpha_numer = numer[k]
            elif opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
                alpha_numer = torch.matmul(cur_weight[j], ref_layer[k].T)
            if opt.version.find('v2') != -1:
                _alpha = alpha_numer / alpha_denom[0][k]
                alpha = deepcopy(_alpha.item())
                del _alpha
            if opt.version in ['v2', 'v2q', 'v2qq', 'v2f']:
                _beta = cur_mean[j][0] - alpha * ref_mean[k][0]
                beta = deepcopy(_beta.item())
                del _beta
            del numer, denom, pcc, abs_pcc

            if opt.version in ['v2', 'v2q', 'v2qq', 'v2f']:
                ref_idx = (k, alpha, beta)
            elif opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
                ref_idx = (k, alpha)
            elif opt.version == 'v1':
                ref_idx = k
            idx.append(ref_idx)
        del cur_weight, cur_length, cur_mean, cur_norm, cur_norm_sq_rt
        idx_all.append(idx)
    del ref_layer, ref_mean, ref_norm, ref_norm_sq, ref_norm_sq_rt

    return idx_all

#TODO: corrcoef 코딩
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

    ref_layer_num = opt.refnum
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
    if opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
        _denom = (ref_layer_slices * ref_layer_slices).sum(dim=1)
    else:
        ref_mean = ref_layer_slices.mean(dim=1, keepdim=True)
        ref_norm = ref_layer_slices - ref_mean
        _denom = (ref_norm * ref_norm).sum(dim=1)

    # add epsilon to every denom
    if opt.version in ['v2q', 'v2qq', 'v2f', 'v2qnb', 'v2qqnb']:
        _denom += opt.epsilon # epsilon for non-zero denom (default: 1e-08)

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        idx = []
        cur_layer = torch.Tensor(w_kernel[i]).cuda()
        # cur_layer = torch.Tensor(w_kernel[i])
        cur_layer = cur_layer.view(cur_layer.size(0), -1)
        cur_layer_length = cur_layer.size(0)
        for j in range(cur_layer_length):
            cur_weight = cur_layer[j].view(-1, pwd)
            cur_length = cur_weight.size(0)
            if opt.version not in ['v2nb', 'v2qnb', 'v2qqnb']:
                cur_mean = cur_weight.mean(dim=1, keepdim=True)
                cur_norm = cur_weight - cur_mean
                numer = torch.matmul(cur_norm, ref_norm.T)
            else:
                numer = torch.matmul(cur_weight, ref_layer_slices.T)
            denom = deepcopy(_denom.expand_as(numer))
            alphas = deepcopy(numer / denom)
            del numer, denom

            if opt.version not in ['v2nb', 'v2qnb', 'v2qqnb']:
                betas = cur_mean - alphas * ref_mean.view(-1, ref_length).expand_as(alphas)
            for idx_cur_slice in range(cur_length):
                cur_alphas = alphas[idx_cur_slice].view(ref_length, -1)
                if opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
                    residual_mat = ref_layer_slices * cur_alphas -\
                        cur_weight[idx_cur_slice].expand_as(ref_layer_slices)
                else:
                    cur_betas = betas[idx_cur_slice].view(ref_length, -1)
                    residual_mat = (ref_layer_slices * cur_alphas + cur_betas) -\
                        cur_weight[idx_cur_slice].expand_as(ref_layer_slices)
                residual_mat = residual_mat.abs().sum(dim=1)
                k = deepcopy(residual_mat.argmin().cpu().item())
                alpha = deepcopy(alphas[idx_cur_slice][k].cpu().item())
                # k = deepcopy(residual_mat.argmin().item())
                # alpha = deepcopy(alphas[idx_cur_slice][k].item())
                if opt.version == 'v1':
                    ref_idx = k
                elif opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
                    ref_idx = (k, alpha)
                else:
                    beta = deepcopy(betas[idx_cur_slice][k].cpu().item())
                    # beta = deepcopy(betas[idx_cur_slice][k].item())
                    ref_idx = (k, alpha, beta)
                idx.append(ref_idx)
            if opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
                del alphas, cur_alphas, cur_weight, cur_length
            else:
                del alphas, betas, cur_alphas, cur_weight, cur_betas, cur_norm, cur_mean, cur_length
            torch.cuda.empty_cache()
        del cur_layer
        torch.cuda.empty_cache()
        idx_all.append(idx)
    if opt.version in ['v2nb', 'v2qnb', 'v2qqnb']:
        del ref_layer, _denom
    else:
        del ref_layer, ref_mean, ref_norm, _denom

    return idx_all


def save_model(ckpt, indices_all):
    r"""Save new model
    """
    if arch_name in hasPWConvArchs:
        indices, indices_pw = indices_all
    else:
        indices = indices_all

    if opt.version in ['v2q', 'v2qq', 'v2f']:
        print('====> {}/{}bit Quantization for alpha/beta...'.format(opt.quant_bit_a, opt.quant_bit_b))
        quantize_ab(indices, num_bits_a=opt.quant_bit_a, num_bits_b=opt.quant_bit_b)
    elif opt.version in ['v2qnb', 'v2qqnb']:
        print('====> {}bit Quantization for alpha...'.format(opt.quant_bit_a))
        quantize_ab(indices, num_bits_a=opt.quant_bit_a)
    if arch_name in hasPWConvArchs:
        if opt.version in ['v2q', 'v2qq', 'v2f']:
            print('====> {}/{}bit Quantization for alpha/beta in pwconv...'.format(opt.quant_bit_a, opt.quant_bit_b))
            quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a, num_bits_b=opt.quant_bit_b)
        elif opt.version in ['v2qnb', 'v2qqnb']:
            print('====> {}bit Quantization for alpha in pwconv...'.format(opt.quant_bit_a))
            quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a)
        indices = (indices, indices_pw)

    ckpt['idx'] = indices
    ckpt['version'] = opt.version
    new_model_filename = '{}_{}'.format(opt.ckpt[:-4], opt.version)
    if arch_name in hasPWConvArchs:
        new_model_filename += '_pwd{}_pws{}'.format(opt.pw_bind_size, opt.pwkernel_stride)
    if opt.version in ['v2qq', 'v2f', 'v2qqnb']:
        new_model_filename += '_q{}a{}'.format(opt.quant_bit, opt.quant_bit_a)
        if opt.version != 'v2qqnb':
            new_model_filename += 'b{}'.format(opt.quant_bit_b)
        new_model_filename += '_eps{:.0e}'.format(opt.epsilon)
    elif opt.version in ['v2q', 'v2qnb']:
        new_model_filename += '_qa{}'.format(opt.quant_bit_a)
        if opt.version != 'v2qnb':
            new_model_filename += 'b{}'.format(opt.quant_bit_b)
        new_model_filename += '_eps{:.0e}'.format(opt.epsilon)
    if opt.version.find('v2') != -1 and opt.ustv1 != '':
        new_model_filename += f'_ustv1-{opt.ustv1}'
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

    ref_layer_num = opt.refnum
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

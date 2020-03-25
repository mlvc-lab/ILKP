import time
import pathlib
import argparse
from os.path import isfile
from tqdm import tqdm

import math
import torch
import numpy as np

import models
from utils import build_model, load_model
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def config():
    parser = argparse.ArgumentParser(description='Find similar kernel')
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
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--groups', default=2, type=int, metavar='N',
                        help='number of groups for ShuffleNet (default: 2)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='Path of checkpoint (Default: none)')
    parser.add_argument('-v', '--version', default='', type=str, metavar='VER',
                        dest='version', help='find kernel version number (default: none)')
    parser.add_argument('-d', '--bind-size', default=2, type=int, metavar='N',
                        dest='bind_size',
                        help='the number of binding channels in convolution '
                             '(subvector size) on version 3 (v3) (default: 2)')
    parser.add_argument('--analysis', dest='analysis', action='store_true',
                        help='Analysis mode?')
    # for quantization
    parser.add_argument('--qb', '--quant_bit', default=8, type=int, metavar='N', dest='quant_bit',
                        help='number of bits for quantization (Default: 8)')

    cfg = parser.parse_args()
    return cfg


def main():
    global opt, dir_path, hasDiffLayersArchs
    opt = config()

    # model
    hasDiffLayersArchs = ['vgg', 'resnet', 'resnext', 'wideresnet']
    if opt.arch in hasDiffLayersArchs:
        print('\n=> creating model \'{}\''.format(opt.arch + str(opt.layers)))
    else:
        print('\n=> creating model \'{}\''.format(opt.arch))

    model = build_model(opt)

    if model is None:
        print('==> unavailable model parameters!! exit...\n')
        exit()

    # checkpoint file
    ckpt_dir = pathlib.Path('checkpoint')
    if opt.arch in hasDiffLayersArchs:
        dir_path = ckpt_dir / (opt.arch + str(opt.layers)) / opt.dataset
    else:
        dir_path = ckpt_dir / opt.arch / opt.dataset
    ckpt_file = dir_path / opt.ckpt

    if isfile(ckpt_file):
        print('==> Loading Checkpoint \'{}\''.format(opt.ckpt))
        checkpoint = load_model(model, ckpt_file,
                                main_gpu=None, use_cuda=False)
        print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                    opt.ckpt, checkpoint['epoch']))

        if opt.analysis:
            if not opt.version == 'v2' or opt.arch in hasDiffLayersArchs:
                print('analysis can only be used with ver2 and dwconv..')
                exit()
            weight_analysis(model, checkpoint)
            return
        if opt.version == 'v2q':
            print('==> {}bit Quantization...'.format(opt.quant_bit))
            quantize(model, opt.quant_bit)
        print('==> Find the most similar kernel in previous layers ' +
              'from filters at Checkpoint \'{}\''.format(opt.ckpt))
        new_ckpt_name = find_kernel(model, checkpoint)
        print('===> Save new Checkpoint \'{}\''.format(new_ckpt_name))
        return
    else:
        print('==> no Checkpoint found at \'{}\''.format(
                    opt.ckpt))
        exit()


def find_kernel(model, ckpt):
    """ find the most similar kernel
    """
    if opt.arch in hasDiffLayersArchs:
        w_conv = model.get_weights_conv(use_cuda=False)
    else:
        w_conv = model.get_weights_dwconv(use_cuda=False)

    start_layer = 1
    ref_layer = 0
    if opt.version in ['v1', 'v2a', 'v3a']:
        start_layer = 2
        ref_layer = 1

    idx_all = []
    if opt.arch in hasDiffLayersArchs:
        if opt.version == 'v3' or opt.version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_conv[ref_layer])):
                for k in range(len(w_conv[ref_layer][j])):
                    concat_kernels_ref.append(w_conv[ref_layer][j][k])
            num_subvec_ref = len(concat_kernels_ref) // d
            for i in range(start_layer, len(w_conv)):
                idx = []
                num_subvec_cur = (len(w_conv[i])*len(w_conv[i][0])) // d
                num_subvec_cur_each_kernel = len(w_conv[i][0]) // d
                for j in tqdm(range(num_subvec_cur), ncols=80, unit='subvector'):
                    subvec_idx_j = j // num_subvec_cur_each_kernel
                    subvec_idx_k = j % num_subvec_cur_each_kernel
                    min_diff = math.inf
                    ref_idx = 0
                    for u in range(num_subvec_ref):
                        mean_cur = np.mean(w_conv[i][subvec_idx_j][d*subvec_idx_k:d*subvec_idx_k+d])
                        mean_ref = np.mean(concat_kernels_ref[d*u:d*(u+1)])
                        alpha_numer = 0.0
                        alpha_denom = 0.0
                        for v in range(d):
                            for row in range(len(concat_kernels_ref[d*u+v])):
                                for col in range(len(concat_kernels_ref[d*u+v][row])):
                                    alpha_numer += ((concat_kernels_ref[d*u+v][row][col] - mean_ref) *
                                                    (w_conv[i][subvec_idx_j][d*subvec_idx_k+v][row][col] - mean_cur))
                                    alpha_denom += ((concat_kernels_ref[d*u+v][row][col] - mean_ref) *
                                                    (concat_kernels_ref[d*u+v][row][col] - mean_ref))
                        alpha = alpha_numer / alpha_denom
                        beta = mean_cur - alpha*mean_ref
                        diff = 0.0
                        for v in range(d):
                            tmp_diff = alpha*concat_kernels_ref[d*u+v]+beta - w_conv[i][subvec_idx_j][d*subvec_idx_k+v]
                            diff += np.sum(np.absolute(tmp_diff))
                        if min_diff > diff:
                            min_diff = diff
                            ref_idx = (u, alpha, beta)
                    idx.append(ref_idx)
                idx_all.append(idx)
        else:
            for i in range(start_layer, len(w_conv)):
                idx = []
                for j in tqdm(range(len(w_conv[i])), ncols=80, unit='filter'):
                    for k in range(len(w_conv[i][j])):
                        min_diff = math.inf
                        ref_idx = 0
                        if opt.version == 'v1':
                            for v in range(len(w_conv[ref_layer])):
                                for w in range(len(w_conv[ref_layer][v])):
                                    diff = np.sum(np.absolute(w_conv[ref_layer][v][w] - w_conv[i][j][k]))
                                    if min_diff > diff:
                                        min_diff = diff
                                        ref_idx = v * len(w_conv[ref_layer]) + w
                        elif opt.version in ['v2', 'v2a', 'v2q']:
                            for v in range(len(w_conv[ref_layer])):
                                for w in range(len(w_conv[ref_layer][v])):
                                    # find alpha, beta using least squared method every kernel in reference layer
                                    mean_cur = np.mean(w_conv[i][j][k])
                                    mean_ref = np.mean(w_conv[ref_layer][v][w])
                                    alpha_numer = 0.0
                                    alpha_denom = 0.0
                                    for row in range(len(w_conv[i][j][k])):
                                        for col in range(len(w_conv[i][j][k][row])):
                                            alpha_numer += ((w_conv[ref_layer][v][w][row][col] - mean_ref) *
                                                            (w_conv[i][j][k][row][col] - mean_cur))
                                            alpha_denom += ((w_conv[ref_layer][v][w][row][col] - mean_ref) *
                                                            (w_conv[ref_layer][v][w][row][col] - mean_ref))
                                    alpha = alpha_numer / alpha_denom
                                    beta = mean_cur - alpha*mean_ref
                                    diff = np.sum(np.absolute(alpha*w_conv[ref_layer][v][w]+beta - w_conv[i][j][k]))
                                    if min_diff > diff:
                                        idxidx = v * len(w_conv[ref_layer]) + w
                                        min_diff = diff
                                        ref_idx = (idxidx, alpha, beta)
                        idx.append(ref_idx)
                idx_all.append(idx)
    else:
        if opt.version == 'v3' or opt.version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_conv[ref_layer])):
                for k in range(len(w_conv[ref_layer][j])):
                    concat_kernels_ref.append(w_conv[ref_layer][j][k])
            num_subvec_ref = len(concat_kernels_ref) // d
            for i in range(start_layer, len(w_conv)):
                idx = []
                num_subvec_cur = (len(w_conv[i])*len(w_conv[i][0])) // d
                for j in tqdm(range(num_subvec_cur), ncols=80, unit='subvector'):
                    min_diff = math.inf
                    ref_idx = 0
                    for u in range(num_subvec_ref):
                        mean_cur = np.mean(w_conv[i][d*j:d*j+d][0])
                        mean_ref = np.mean(concat_kernels_ref[d*u:d*(u+1)])
                        alpha_numer = 0.0
                        alpha_denom = 0.0
                        for v in range(d):
                            for row in range(len(concat_kernels_ref[d*u+v])):
                                for col in range(len(concat_kernels_ref[d*u+v][row])):
                                    alpha_numer += ((concat_kernels_ref[d*u+v][row][col] - mean_ref) *
                                                    (w_conv[i][d*j+v][0][row][col] - mean_cur))
                                    alpha_denom += ((concat_kernels_ref[d*u+v][row][col] - mean_ref) *
                                                    (concat_kernels_ref[d*u+v][row][col] - mean_ref))
                        alpha = alpha_numer / alpha_denom
                        beta = mean_cur - alpha*mean_ref
                        diff = 0.0
                        for v in range(d):
                            tmp_diff = alpha*concat_kernels_ref[d*u+v]+beta - w_conv[i][d*j+v][0]
                            diff += np.sum(np.absolute(tmp_diff))
                        if min_diff > diff:
                            min_diff = diff
                            ref_idx = (u, alpha, beta)
                    idx.append(ref_idx)
                idx_all.append(idx)
        else:
            for i in range(start_layer, len(w_conv)):
                idx = []
                for j in tqdm(range(len(w_conv[i])), ncols=80, unit='filter'):
                    min_diff = math.inf
                    ref_idx = 0
                    if opt.version == 'v1':
                        for k in range(len(w_conv[ref_layer])):
                            diff = np.sum(np.absolute(w_conv[ref_layer][k][0] - w_conv[i][j][0]))
                            if min_diff > diff:
                                min_diff = diff
                                ref_idx = k
                    elif opt.version in ['v2', 'v2a', 'v2q']:
                        for k in range(len(w_conv[ref_layer])):
                            # find alpha, beta using least squared method every kernel in reference layer
                            mean_cur = np.mean(w_conv[i][j][0])
                            mean_ref = np.mean(w_conv[ref_layer][k][0])
                            alpha_numer = 0.0
                            alpha_denom = 0.0
                            for u in range(3):
                                for v in range(3):
                                    alpha_numer += ((w_conv[ref_layer][k][0][u][v] - mean_ref) *
                                                    (w_conv[i][j][0][u][v] - mean_cur))
                                    alpha_denom += ((w_conv[ref_layer][k][0][u][v] - mean_ref) *
                                                    (w_conv[ref_layer][k][0][u][v] - mean_ref))
                            alpha = alpha_numer / alpha_denom
                            beta = mean_cur - alpha*mean_ref
                            diff = np.sum(np.absolute(alpha*w_conv[ref_layer][k][0]+beta - w_conv[i][j][0]))
                            if min_diff > diff:
                                min_diff = diff
                                ref_idx = (k, alpha, beta)
                    idx.append(ref_idx)
                idx_all.append(idx)

    ckpt['idx'] = idx_all
    ckpt['version'] = opt.version

    if opt.version == 'v3' or opt.version == 'v3a':
        new_model_filename = '{}_{}_d{}.pth'.format(opt.ckpt[:-4], opt.version, opt.bind_size)
    elif opt.version == 'v2q':
        new_model_filename = '{}_{}_q{}.pth'.format(opt.ckpt[:-4], opt.version, opt.quant_bit)
    else:
        new_model_filename = '{}_{}.pth'.format(opt.ckpt[:-4], opt.version)
    model_file = dir_path / new_model_filename
    torch.save(ckpt, model_file)

    return new_model_filename
    # TODO
    # 저장을 어떻게 할 것인가
    # 원래 코드에서 model.state_dict()만 저장하고 불러들이는데 사용됨.
    # conv weight만 따로 빼서 저장해보기
    # 새로운 방법은 index 따로 weight 따로 저장.
    # main.py에서 불러들이는 거 구현


def weight_analysis(model, ckpt):
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dir_weights = pathlib.Path('weights')
    dir_weights.mkdir(parents=True, exist_ok=True)

    w_conv = model.get_weights_dwconv(use_cuda=False)
    start_layer = 1
    ref_layer = 0
    idx_all = []
    for i in range(start_layer, len(w_conv)):
        idx = []
        for j in tqdm(range(len(w_conv[i])), ncols=80, unit='filter'):
            min_diff = math.inf
            ref_idx = 0
            for k in range(len(w_conv[ref_layer])):
                # find alpha, beta using least squared method every kernel in reference layer
                mean_cur = np.mean(w_conv[i][j][0])
                mean_ref = np.mean(w_conv[ref_layer][k][0])
                alpha_numer = 0.0
                alpha_denom = 0.0
                for u in range(3):
                    for v in range(3):
                        alpha_numer += ((w_conv[ref_layer][k][0][u][v] - mean_ref) *
                                        (w_conv[i][j][0][u][v] - mean_cur))
                        alpha_denom += ((w_conv[ref_layer][k][0][u][v] - mean_ref) *
                                        (w_conv[ref_layer][k][0][u][v] - mean_ref))
                alpha = alpha_numer / alpha_denom
                beta = mean_cur - alpha*mean_ref
                diff = np.sum(np.absolute(alpha*w_conv[ref_layer][k][0]+beta - w_conv[i][j][0]))
                if min_diff > diff:
                    min_diff = diff
                    ref_idx = k
            idx.append(ref_idx)
        idx_all.append(idx)
    layerNumList = list(range(start_layer, len(w_conv)))
    randLayerIdx = random.sample(layerNumList,3)
    for i in randLayerIdx:
        kernelNumList = list(range(len(w_conv[i])))
        randKernelIdx = random.sample(kernelNumList,3)
        for j in randKernelIdx:
            k = idx_all[i-start_layer][j]
            weights_cur = []
            weights_ref = []
            for u in range(3):
                for v in range(3):
                    weights_cur.append(w_conv[i][j][0][u][v])
                    weights_ref.append(w_conv[ref_layer][k][0][u][v])
            plt.figure(figsize=(8,6), dpi=300)
            plt.title('weights')
            plt.xlabel('current kernel K_{},{}'.format(i,j))
            plt.ylabel('reference kernel K_{},{}'.format(0,k))
            plt.scatter(weights_cur, weights_ref)
            plt.savefig(dir_weights/'{}_{}_weight_{}_{}.png'.format(opt.arch, opt.dataset, i, j),
                        bbox_inches='tight')


def quantize(model, num_bits=8):
    """quantize weights of convolution kernels
    """
    if opt.arch in hasDiffLayersArchs:
        w_conv = model.get_weights_conv(use_cuda=False)
    else:
        w_conv = model.get_weights_dwconv(use_cuda=False)

    num_layer = len(w_conv)

    qmin = -2.**(num_bits - 1.)
    qmax = 2.**(num_bits - 1.) - 1.

    for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
        min_val = np.amin(w_conv[i])
        max_val = np.amax(w_conv[i])
        scale = (max_val - min_val) / (qmax - qmin)
        w_conv[i] = np.around(np.clip(w_conv[i] / scale, qmin, qmax))
        w_conv[i] = scale * w_conv[i]

    if opt.arch in hasDiffLayersArchs:
        model.set_weights_conv(w_conv, use_cuda=False)
    else:
        model.set_weights_dwconv(w_conv, use_cuda=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("====> total time: {:.2f}s".format(elapsed_time))

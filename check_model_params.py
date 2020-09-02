import argparse
import numpy as np
import models
from utils import set_arch_name

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def config():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='Check model parameters')
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
    # for calculating number of pwkernel slice
    parser.add_argument('-pwd', '--pw-bind-size', default=8, type=int, metavar='N',
                        dest='pw_bind_size',
                        help='the number of binding channels in pointwise convolution '
                             '(subvector size) (default: 8)')
    cfg = parser.parse_args()
    return cfg


def main():
    opt = config()

    # set model name
    arch_name = set_arch_name(opt)

    # calculate number of pwkernel slice
    # model = models.__dict__[opt.arch](data='cifar10', num_layers=opt.layers,
    #                                   width_mult=opt.width_mult, batch_norm=opt.bn)
    # w_kernel = model.get_weights_conv(use_cuda=False)
    # for i in range(len(w_kernel)):
    #     print(np.shape(w_kernel[i]))
    # w_pwkernel = model.get_weights_pwconv(use_cuda=False)
    # d = opt.pw_bind_size
    # sum_slices = 0
    # sum_num_weights = 0
    # for i in range(len(w_pwkernel)):
    #     c_out, c_in, _, _ = np.shape(w_pwkernel[i])
    #     num_weights = c_out * c_in
    #     sum_num_weights += num_weights
    #     if i == 0:
    #         num_slice = c_out * (c_in - d + 1)
    #     else:
    #         num_slice = c_out * (c_in // d)
    #         sum_slices += num_slice
    #     print('[{}-th layer]  #weights: {} / #slices: {}'.format(i, num_weights, num_slice))
    # print('\ntotal #weights: {} / total #slices (except ref_layer): {}'.format(sum_num_weights, sum_slices))

    print('\n[ {}-cifar10 parameters ]'.format(arch_name))
    model = models.__dict__[opt.arch](data='cifar10', num_layers=opt.layers,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)
    # for name, param in model.named_parameters():
    #     if name.find('linear') != -1:
    #         print('{}: {}'.format(name, param.numel()))
    # for name, param in model.named_parameters():
    #     print('{}: {}'.format(name, param.numel()))
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of all parameters: ', num_params)
    print('Number of all trainable parameters: ', num_trainable_params)

    print('\n[ {}-cifar100 parameters ]'.format(arch_name))
    model = models.__dict__[opt.arch](data='cifar100', num_layers=opt.layers,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)
    # for name, param in model.named_parameters():
    #     if name.find('linear') != -1:
    #         print('{}: {}'.format(name, param.numel()))
    # for name, param in model.named_parameters():
    #     print('{}: {}'.format(name, param.numel()))
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of all parameters: ', num_params)
    print('Number of trainable parameters: ', num_trainable_params)

    # print('\n[ {}-imagenet parameters ]'.format(arch_name))
    # model = models.__dict__[opt.arch](data='imagenet', num_layers=opt.layers,
    #                                   width_mult=opt.width_mult, batch_norm=opt.bn)
    # for name, param in model.named_parameters():
    #     if name.find('linear') != -1:
    #         print('{}: {}'.format(name, param.numel()))
    # for name, param in model.named_parameters():
    #     print('{}: {}'.format(name, param.numel()))
    # num_params = sum(p.numel() for p in model.parameters())
    # num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Number of all parameters: ', num_params)
    # print('Number of trainable parameters: ', num_trainable_params)


if __name__ == '__main__':
    main()

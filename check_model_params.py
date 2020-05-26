import argparse
import models
from utils import hasDiffLayersArchs

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
    cfg = parser.parse_args()
    return cfg


def main():
    opt = config()

    # set model name
    arch_name = opt.arch
    if opt.arch in hasDiffLayersArchs:
        arch_name += str(opt.layers)

    print('\n[ {}-cifar10 parameters ]'.format(arch_name))
    model = models.__dict__[opt.arch](data='cifar10', num_layers=opt.layers,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)
    # for name, param in model.named_parameters():
    #     if name.find('linear') != -1:
    #         print('{}: {}'.format(name, param.numel()))
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, param.numel()))
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
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, param.numel()))
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of all parameters: ', num_params)
    print('Number of trainable parameters: ', num_trainable_params)

    print('\n[ {}-imagenet parameters ]'.format(arch_name))
    model = models.__dict__[opt.arch](data='imagenet', num_layers=opt.layers,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)
    # for name, param in model.named_parameters():
    #     if name.find('linear') != -1:
    #         print('{}: {}'.format(name, param.numel()))
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, param.numel()))
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of all parameters: ', num_params)
    print('Number of trainable parameters: ', num_trainable_params)


if __name__ == '__main__':
    main()

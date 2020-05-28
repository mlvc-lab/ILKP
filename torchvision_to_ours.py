'''
Usage:
    python3 torchvision_to_ours.py -a resnet --layers 152 -C -g 0 -E
'''
import time
import pathlib
from os.path import isfile

import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import build_model, AverageMeter, ProgressMeter, accuracy, set_arch_name
from config import config
from data import DataLoader

# for ignore imagenet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


model_urls = {
    'resnet18':  'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':  'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':  'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'wideresnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wideresnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def config():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='Change torchvision checkpoint to ours')
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
    parser.add_argument('-E', '--evaluate', dest='evaluate', action='store_true',
                        help='Test model?')
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='Use cuda?')
    parser.add_argument('-g', '--gpuids', default=[0], nargs='+',
                        help='GPU IDs for using (Default: 0)')
    cfg = parser.parse_args()
    cfg.gpuids = list(map(int, cfg.gpuids))
    return cfg


def main():
    global opt
    opt = config()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # set model name
    arch_name = set_arch_name(opt)

    dir_ckpt = pathlib.Path('checkpoint')
    dir_path = dir_ckpt / arch_name / opt.dataset
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.utils.model_zoo.load_url(model_urls[arch_name], dir_path.as_posix())

    print('\n=> creating model \'{}\''.format(arch_name))
    model = build_model(opt)

    if model is None:
        print('==> unavailable model parameters!! exit...\n')
        exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=opt.momentum, weight_decay=opt.weight_decay,
                          nesterov=True)
    start_epoch = 0

    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids,
                                output_device=opt.gpuids[0])
        cudnn.benchmark = True

    # checkpoint file
    ckpt_file = dir_path / model_urls[arch_name][36:]


    # for evaluation
    if opt.evaluate:
        if isfile(ckpt_file):
            # Data loading
            print('==> Load data..')
            train_loader, val_loader = DataLoader(opt.batch_size, opt.workers,
                                                  opt.dataset, opt.datapath,
                                                  opt.cuda)
            # Checkpoint loading
            print('==> Loading Checkpoint...')
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=opt.gpuids[0], use_cuda=opt.cuda)
            print('==> Loaded Checkpoint...')

            # evaluate on validation set
            print('\n===> [ Evaluation ]')
            acc1, acc5 = validate(val_loader, model, criterion)
            best_acc1 = max(acc1_valid, best_acc1)
            print('==> Saving Checkpoint...')
            state = {'epoch': 1,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'n_retrain': 0,
                     'new': False}
            torch.save(state, dir_path / 'ckpt_best_torchvision.pth')
            print('==> Saved Checkpoint at ckpt_best_torchvision.pth')
            return
        else:
            print('==> no checkpoint found...')
            exit()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if opt.cuda:
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % opt.print_freq == 0:
                progress.print(i)

            end = time.time()

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def load_model(model, ckpt_file, main_gpu, use_cuda=True):
    if use_cuda:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(main_gpu))
        try:
            model.load_state_dict(checkpoint)
        except:
            model.module.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint)
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k[:]
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

    return checkpoint
    

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))

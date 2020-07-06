import time
import pathlib
from os.path import isfile

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
import config
from utils import *
from data import DataLoader
from find_similar_kernel import find_kernel, find_kernel_pw
from quantize import quantize, quantize_ab

# for ignore imagenet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# for sacred logging
from sacred import Experiment
from sacred.observers import MongoObserver

# sacred experiment
ex = Experiment('WP_MESS')
ex.observers.append(MongoObserver.create(url=config.MONGO_URI,
                                         db_name=config.MONGO_DB))


@ex.config
def hyperparam():
    """
    sacred exmperiment hyperparams
    :return:
    """
    args = config.config()


@ex.main
def main(args):
    global arch_name
    opt = args
    if opt.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # set model name
    arch_name = set_arch_name(opt)

    # logging at sacred
    ex.log_scalar('architecture', arch_name)
    ex.log_scalar('dataset', opt.dataset)

    print('\n=> creating model \'{}\''.format(arch_name))
    model = models.__dict__[opt.arch](data=opt.dataset, num_layers=opt.layers,
                                      width_mult=opt.width_mult, batch_norm=opt.bn)

    if model is None:
        print('==> unavailable model parameters!! exit...\n')
        exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=opt.momentum, weight_decay=opt.weight_decay,
                          nesterov=True)
    start_epoch = 0
    n_retrain = 0

    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids,
                                output_device=opt.gpuids[0])
        cudnn.benchmark = True

    # checkpoint file
    ckpt_dir = pathlib.Path('checkpoint')
    ckpt_file = ckpt_dir / arch_name / opt.dataset / opt.ckpt

    # for resuming training
    if opt.resume:
        if isfile(ckpt_file):
            print('==> Loading Checkpoint \'{}\''.format(opt.ckpt))
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=opt.gpuids[0], use_cuda=opt.cuda)

            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                opt.ckpt, start_epoch))
        else:
            print('==> no checkpoint found \'{}\''.format(
                opt.ckpt))
            exit()

    # Data loading
    print('==> Load data..')
    start_time = time.time()
    train_loader, val_loader = DataLoader(opt.batch_size, opt.workers,
                                          opt.dataset, opt.datapath,
                                          opt.cuda)
    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time//60), elapsed_time%60))
    print('===> Data loaded..')

    # for evaluation
    if opt.evaluate:
        if isfile(ckpt_file):
            print('==> Loading Checkpoint \'{}\''.format(opt.ckpt))
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=opt.gpuids[0], use_cuda=opt.cuda)
            epoch = checkpoint['epoch']
            # logging at sacred
            ex.log_scalar('best_epoch', epoch)

            if opt.new:
                # logging at sacred
                ex.log_scalar('version', checkpoint['version'])
                # if checkpoint['version'] in ['v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
                #     ex.log_scalar('epsilon', opt.epsilon)
                print('===> Change indices to weights..')
                idxtoweight(opt, model, checkpoint['idx'], checkpoint['version'])

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                opt.ckpt, epoch))

            # evaluate on validation set
            print('\n===> [ Evaluation ]')
            start_time = time.time()
            acc1, acc5 = validate(opt, val_loader, None, model, criterion)
            elapsed_time = time.time() - start_time
            acc1 = round(acc1.item(), 4)
            acc5 = round(acc5.item(), 4)
            ckpt_name = '{}-{}-{}'.format(arch_name, opt.dataset, opt.ckpt[:-4])
            save_eval([ckpt_name, acc1, acc5])
            print('====> {:.2f} seconds to evaluate this model\n'.format(
                elapsed_time))
            return acc1
        else:
            print('==> no checkpoint found \'{}\''.format(
                opt.ckpt))
            exit()

    # for retraining
    if opt.retrain:
        if isfile(ckpt_file):
            print('==> Loading Checkpoint \'{}\''.format(opt.ckpt))
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=opt.gpuids[0], use_cuda=opt.cuda)
            try:
                n_retrain = checkpoint['n_retrain'] + 1
            except:
                n_retrain = 1

            # logging at sacred
            ex.log_scalar('n_retrain', n_retrain)

            if not opt.quant:
                if opt.version != checkpoint['version']:
                    print('version argument is different with saved checkpoint version!!')
                    exit()

                # logging at sacred
                ex.log_scalar('version', checkpoint['version'])
                # if checkpoint['version'] in ['v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
                #     ex.log_scalar('epsilon', opt.epsilon)

                print('===> Change indices to weights..')
                idxtoweight(opt, model, checkpoint['idx'], opt.version)

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                opt.ckpt, checkpoint['epoch']))
        else:
            print('==> no checkpoint found \'{}\''.format(
                opt.ckpt))
            exit()

    # train...
    best_acc1 = 0.0
    train_time = 0.0
    validate_time = 0.0
    extra_time = 0.0
    for epoch in range(start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch, opt)
        if not opt.new:
            print('\n==> {}/{} training'.format(
                arch_name, opt.dataset))
        else:
            print('\n==> {}/{}-new_{} training'.format(
                arch_name, opt.dataset, opt.version))
            print('==> Version: {} / SaveEpoch: {}'.format(
                opt.version, opt.save_epoch))
        print('==> Epoch: {}, lr = {}'.format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        print('===> [ Training ]')
        start_time = time.time()
        acc1_train, acc5_train = train(opt, train_loader,
            epoch=epoch, model=model,
            criterion=criterion, optimizer=optimizer)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print('====> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))
        start_time = time.time()
        if opt.new:
            # every 5 epochs
            if (epoch+1) % opt.save_epoch == 0:
                print('===> Change kernels using {}'.format(opt.version))
                indices = find_similar_kernel_n_change(opt, model, opt.version)
        elapsed_time = time.time() - start_time
        extra_time += elapsed_time
        print('====> {:.2f} seconds for extra time this epoch\n'.format(
            elapsed_time))

        # evaluate on validation set
        print('===> [ Validation ]')
        start_time = time.time()
        acc1_valid, acc5_valid = validate(opt, val_loader, epoch, model, criterion)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print('====> {:.2f} seconds to validate this epoch\n'.format(
            elapsed_time))

        acc1_train = round(acc1_train.item(), 4)
        acc5_train = round(acc5_train.item(), 4)
        acc1_valid = round(acc1_valid.item(), 4)
        acc5_valid = round(acc5_valid.item(), 4)

        # remember best Acc@1 and save checkpoint and summary csv file
        state = {'epoch': epoch + 1,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'n_retrain': n_retrain}
        summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

        if not opt.new:
            state['new'] = False
            state['version'] = ''
            state['idx'] = []
            is_best = acc1_valid > best_acc1
            best_acc1 = max(acc1_valid, best_acc1)
            save_model(arch_name, state, epoch, is_best, opt, n_retrain)
            save_summary(arch_name, summary, opt, n_retrain)
        else:
            # every 'opt.save_epoch' epochs
            if (epoch+1) % opt.save_epoch == 0:
                state['new'] = True
                state['version'] = opt.version
                state['idx'] = indices
                is_best = acc1_valid > best_acc1
                best_acc1 = max(acc1_valid, best_acc1)
                save_model(arch_name, state, epoch, is_best, opt, n_retrain)
                save_summary(arch_name, summary, opt, n_retrain)

    # calculate time 
    avg_train_time = train_time / (opt.epochs - start_epoch)
    avg_valid_time = validate_time / (opt.epochs - start_epoch)
    avg_extra_time = extra_time / (opt.epochs - start_epoch)
    total_train_time = train_time + validate_time + extra_time
    print('====> average training time each epoch: {:,}m {:.2f}s'.format(
        int(avg_train_time//60), avg_train_time%60))
    print('====> average validation time each epoch: {:,}m {:.2f}s'.format(
        int(avg_valid_time//60), avg_valid_time%60))
    print('====> average extra time each epoch: {:,}m {:.2f}s'.format(
        int(avg_extra_time//60), avg_extra_time%60))
    print('====> training time: {}h {}m {:.2f}s'.format(
        int(train_time//3600), int((train_time%3600)//60), train_time%60))
    print('====> validation time: {}h {}m {:.2f}s'.format(
        int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
    print('====> extra time: {}h {}m {:.2f}s'.format(
        int(extra_time//3600), int((extra_time%3600)//60), extra_time%60))
    print('====> total training time: {}h {}m {:.2f}s'.format(
        int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))

    return best_acc1


def train(opt, train_loader, **kwargs):
    r"""Train model each epoch
    """
    epoch = kwargs.get('epoch')
    model = kwargs.get('model')
    criterion = kwargs.get('criterion')
    optimizer = kwargs.get('optimizer')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt.cuda:
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        # option 1) add inverse nuclear norm loss
        if opt.nuc_loss:
            regularizer = new_regularizer(opt, model, 'nuc')
            loss += regularizer
        # option 2) add absolute pcc loss
        if opt.pcc_loss:
            regularizer = new_regularizer(opt, model, 'pcc')
            loss += regularizer

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % opt.print_freq == 0:
            progress.print(i)

        end = time.time()

    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # logging at sacred
    ex.log_scalar('train.loss', losses.avg, epoch)
    ex.log_scalar('train.top1', top1.avg.item(), epoch)
    ex.log_scalar('train.top5', top5.avg.item(), epoch)

    return top1.avg, top5.avg


def validate(opt, val_loader, epoch, model, criterion):
    r"""Validate model each epoch and evaluation
    """
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

    # logging at sacred
    ex.log_scalar('test.loss', losses.avg, epoch)
    ex.log_scalar('test.top1', top1.avg.item(), epoch)
    ex.log_scalar('test.top5', top5.avg.item(), epoch)

    return top1.avg, top5.avg


def new_regularizer(opt, model, regularizer_name='nuc'):
    r"""Add new regularizer

    Args:
        regularizer_name (str): name of regularizer
    """
    d = opt.bind_size
    if opt.arch in hasDiffLayersArchs:
        try:
            first_conv = model.module.get_layer_conv(0).weight
        except:
            first_conv = model.get_layer_conv(0).weight
    else:
        try:
            first_conv = model.module.get_layer_dwconv(0).weight
        except:
            first_conv = model.get_layer_dwconv(0).weight
    first_conv = torch.flatten(first_conv, start_dim=0, end_dim=1)

    if regularizer_name == 'nuc':
        for idx in range(0, first_conv.size()[0], d):
            sub_tensor = torch.unsqueeze(torch.flatten(first_conv[idx:idx+d]), 0)
            if idx == 0:
                first_conv_all = sub_tensor
            else:
                first_conv_all = torch.cat((first_conv_all, sub_tensor), 0)
        regularizer = opt.nls / torch.norm(first_conv_all, p='nuc')
    elif regularizer_name == 'pcc':
        sub_tensors = []
        for idx in range(0, first_conv.size()[0], d):
            sub_tensors.append(torch.flatten(first_conv[idx:idx+d]))
        sum_abspcc = 0.0
        for idx_x in range(len(sub_tensors)):
            for idx_y in range(idx_x+1,len(sub_tensors)):
                cov_xy = 0.0
                stddev_x = 0.0
                stddev_y = 0.0
                mean_x = torch.mean(sub_tensors[idx_x])
                mean_y = torch.mean(sub_tensors[idx_y])
                for idx_i in range(len(sub_tensors[idx_x])):
                    cov_xy += (sub_tensors[idx_x][idx_i] - mean_x)*(sub_tensors[idx_y][idx_i] - mean_y)
                    stddev_x += (sub_tensors[idx_x][idx_i] - mean_x)*(sub_tensors[idx_x][idx_i] - mean_x)
                    stddev_y += (sub_tensors[idx_y][idx_i] - mean_y)*(sub_tensors[idx_y][idx_i] - mean_y)
                stddev_x = torch.sqrt(stddev_x)
                stddev_y = torch.sqrt(stddev_y)
                pcc_xy = cov_xy / (stddev_x*stddev_y)
                sum_abspcc += torch.abs(pcc_xy)
        regularizer = opt.pls * sum_abspcc

    return regularizer


#TODO: v2f k fix하고 alpha beta 찾는 방법 코딩
def find_similar_kernel_n_change(opt, model, version):
    r"""Find the most similar kernel and change the kernel

    Args:
        version (str): version name of new method
    """
    indices = find_kernel(model, opt)
    if arch_name in hasPWConvArchs and not opt.np:
        indices_pw = find_kernel_pw(model, opt)

    # if version in ['v2qq', 'v2f', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
    #     print('====> {}/{}bit Quantization for alpha/beta...'.format(opt.quant_bit_a, opt.quant_bit_b))
    #     quantize_ab(indices, num_bits_a=opt.quant_bit_a, num_bits_b=opt.quant_bit_b)
    # elif version == 'v2nb':
    #     print('====> {}bit Quantization for alpha...'.format(opt.quant_bit_a))
    #     quantize_ab(indices, num_bits_a=opt.quant_bit_a)
    if arch_name in hasPWConvArchs and not opt.np:
        # if version in ['v2qq', 'v2f', 'v2qq-epsv1', 'v2qq-epsv2', 'v2qq-epsv3']:
        #     print('====> {}/{}bit Quantization for alpha/beta in pwconv...'.format(opt.quant_bit_a, opt.quant_bit_b))
        #     quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a, num_bits_b=opt.quant_bit_b)
        # elif version == 'v2nb':
        #     print('====> {}bit Quantization for alpha in pwconv...'.format(opt.quant_bit_a))
        #     quantize_ab(indices_pw, num_bits_a=opt.quant_bit_a)
        indices = (indices, indices_pw)

    # change idx to kernel
    print('===> Change indices to weights..')
    idxtoweight(opt, model, indices, version)

    return indices


def idxtoweight(opt, model, indices_all, version):
    r"""Change indices to weights

    Args:
        indices_all (list): all indices with index of the most similar kernel, $\alpha$ and $\beta$
        version (str): version name of new method
    """
    w_kernel = get_kernel(model, opt)
    num_layer = len(w_kernel)
    if arch_name in hasPWConvArchs and not opt.np:
        w_pwkernel = get_kernel(model, opt, is_pw=True)
        num_pwlayer = len(w_pwkernel)

    if arch_name in hasPWConvArchs and not opt.np:
        indices, indices_pw = indices_all
    else:
        indices = indices_all

    ref_layer_num = 0
    if version.find('v2') != -1:
        for i in tqdm(range(1, num_layer), ncols=80, unit='layer'):
            for j in range(len(w_kernel[i])):
                for k in range(len(w_kernel[i][j])):
                    ref_idx, alpha, beta = indices[i-1][j*len(w_kernel[i][j])+k]
                    v = ref_idx // len(w_kernel[ref_layer_num][0])
                    w = ref_idx % len(w_kernel[ref_layer_num][0])
                    w_kernel[i][j][k] = alpha * w_kernel[ref_layer_num][v][w] + beta

    if arch_name in hasPWConvArchs and not opt.np:
        if version.find('v2') != -1:
            pwd = opt.pw_bind_size
            pws = opt.pwkernel_stride
            ref_layer = torch.Tensor(w_pwkernel[ref_layer_num])
            ref_layer = ref_layer.view(ref_layer.size(0), ref_layer.size(1))
            ref_layer_slices = None
            num_slices = (ref_layer.size(1) - pwd) // pws + 1
            for i in range(0, ref_layer.size(1) - pwd + 1, pws):
                if ref_layer_slices == None:
                    ref_layer_slices = ref_layer[:,i:i+pwd]
                else:
                    ref_layer_slices = torch.cat((ref_layer_slices, ref_layer[:,i:i+pwd]), dim=1)
            if ((ref_layer.size(1) - pwd) % pws) != 0:
                ref_layer_slices = torch.cat((ref_layer_slices, ref_layer[:, -pwd:]), dim=1)
                num_slices += 1
            ref_layer_slices = ref_layer_slices.view(ref_layer.size(0)*num_slices, pwd)
            ref_layer_slices = ref_layer_slices.view(-1, pwd, 1, 1).numpy()
            for i in tqdm(range(1, num_pwlayer), ncols=80, unit='layer'):
                for j in range(len(w_pwkernel[i])):
                    num_slices = len(w_pwkernel[i][j])//pwd
                    for k in range(num_slices):
                        ref_idx, alpha, beta = indices_pw[i-1][j*num_slices+k]
                        w_pwkernel[i][j][k*pwd:(k+1)*pwd] = alpha * ref_layer_slices[ref_idx] + beta

    set_kernel(w_kernel, model, opt)
    if arch_name in hasPWConvArchs and not opt.np:
        set_kernel(w_pwkernel, model, opt, is_pw=True)


if __name__ == '__main__':
    start_time = time.time()
    ex.run()
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))

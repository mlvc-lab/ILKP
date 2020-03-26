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

from utils import *
from config import config
from data import DataLoader

# for ignore imagenet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def main():
    global opt, hasDiffLayersArchs
    opt = config()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    hasDiffLayersArchs = ['vgg', 'resnet', 'resnext', 'wideresnet']
    if opt.arch in hasDiffLayersArchs:
        print('\n=> creating model \'{}\''.format(opt.arch + str(opt.layers)))
    else:
        print('\n=> creating model \'{}\''.format(opt.arch))

    model = build_model(opt)

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
    if opt.arch in hasDiffLayersArchs:
        ckpt_file = ckpt_dir / (opt.arch + str(opt.layers)) / opt.dataset / opt.ckpt
    else:
        ckpt_file = ckpt_dir / opt.arch / opt.dataset / opt.ckpt

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
    train_loader, val_loader = DataLoader(opt.batch_size, opt.workers,
                                          opt.dataset, opt.datapath,
                                          opt.cuda)

    # for evaluation
    if opt.evaluate:
        if isfile(ckpt_file):
            print('==> Loading Checkpoint \'{}\''.format(opt.ckpt))
            checkpoint = load_model(model, ckpt_file,
                                    main_gpu=opt.gpuids[0], use_cuda=opt.cuda)
            start_epoch = checkpoint['epoch']

            if opt.new:
                print('===> Change indices to weights..')
                idxtoweight(model, checkpoint['idx'], checkpoint['version'])

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                opt.ckpt, start_epoch))

            # evaluate on validation set
            print('\n===> [ Evaluation ]')
            start_time = time.time()
            acc1, acc5 = validate(val_loader, model, criterion)
            if opt.arch in hasDiffLayersArchs:
                ckpt_name = '{}-{}-{}'.format(opt.arch+str(opt.layers), opt.dataset, opt.ckpt[:-4])
            else:
                ckpt_name = '{}-{}-{}'.format(opt.arch, opt.dataset, opt.ckpt[:-4])
            save_eval([ckpt_name, str(acc1)[7:-18], str(acc5)[7:-18]])
            elapsed_time = time.time() - start_time
            print('====> {:.2f} seconds to evaluate this model\n'.format(
                elapsed_time))
            return
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

            if not opt.quant:
                version = checkpoint['version']
                if opt.version != version:
                    print('version argument is different with saved checkpoint version!')
                    exit()

                print('===> Change indices to weights..')
                idxtoweight(model, checkpoint['idx'], version)

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
    for epoch in range(start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch, opt.lr)
        if opt.retrain:
            if opt.new:
                if opt.arch in hasDiffLayersArchs:
                    if version == 'v2q':
                        print('\n==> {}/{} {}-th {}bit retraining'.format(
                            opt.arch+str(opt.layers), opt.dataset, n_retrain, opt.quant_bit))
                    else:
                        print('\n==> {}/{} {}-th retraining'.format(
                            opt.arch+str(opt.layers), opt.dataset, n_retrain))
                else:
                    if version == 'v2q':
                        print('\n==> {}/{} {}-th {}bit retraining'.format(
                            opt.arch, opt.dataset, n_retrain, opt.quant_bit))
                    else:
                        print('\n==> {}/{} {}-th retraining'.format(
                            opt.arch, opt.dataset, n_retrain))
            else:
                if opt.quant:
                    if opt.arch in hasDiffLayersArchs:
                        print('\n==> {}/{} {}-th {}bit retraining'.format(
                            opt.arch+str(opt.layers), opt.dataset, n_retrain, opt.quant_bit))
                    else:
                        print('\n==> {}/{} {}-th {}bit retraining'.format(
                            opt.arch, opt.dataset, n_retrain, opt.quant_bit))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))
            # train for one epoch
            print('===> [ Retraining ]')
            start_time = time.time()
            acc1_train, acc5_train = train(train_loader,
                epoch=epoch, model=model,
                criterion=criterion, optimizer=optimizer)
            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))
            if opt.new:
                if version == 'v2q':
                    print('==> {}bit Quantization...'.format(opt.quant_bit))
                    quantize(model, opt.quant_bit)
                # every 'opt.save_epoch' epochs
                if (epoch+1) % opt.save_epoch == 0:
                    print('===> Change kernels using {}'.format(version))
                    idx = find_similar_kernel_n_change(model, version)
            else:
                if opt.quant:
                    print('==> {}bit Quantization...'.format(opt.quant_bit))
                    quantize(model, opt.quant_bit)
        else:
            if not opt.new:
                if opt.arch in hasDiffLayersArchs:
                    print('\n==> {}/{} training'.format(
                        opt.arch+str(opt.layers), opt.dataset))
                else:
                    print('\n==> {}/{} training'.format(
                        opt.arch, opt.dataset))
                print('==> Epoch: {}, lr = {}'.format(
                    epoch, optimizer.param_groups[0]["lr"]))
            else:
                if opt.arch in hasDiffLayersArchs:
                    if opt.version == 'v3' or opt.version == 'v3a':
                        print('\n==> {}/{}-new_{}_d{} training'.format(
                            opt.arch+str(opt.layers), opt.dataset, opt.version, opt.bind_size))
                    elif opt.version == 'v2q':
                        print('\n==> {}/{}-new_{} {}bit training'.format(
                            opt.arch+str(opt.layers), opt.dataset, opt.version, opt.quant_bit))
                    else:
                        print('\n==> {}/{}-new_{} training'.format(
                            opt.arch+str(opt.layers), opt.dataset, opt.version))
                else:
                    if opt.version == 'v3' or opt.version == 'v3a':
                        print('\n==> {}/{}-new_{}_d{} training'.format(
                            opt.arch, opt.dataset, opt.version, opt.bind_size))
                    elif opt.version == 'v2q':
                        print('\n==> {}/{}-new_{} {}bit training'.format(
                            opt.arch, opt.dataset, opt.version, opt.quant_bit))
                    else:
                        print('\n==> {}/{}-new_{} training'.format(
                            opt.arch, opt.dataset, opt.version))
                print('==> Epoch: {}, lr = {}'.format(
                    epoch, optimizer.param_groups[0]["lr"]))
            # train for one epoch
            print('===> [ Training ]')
            start_time = time.time()
            acc1_train, acc5_train = train(train_loader,
                epoch=epoch, model=model,
                criterion=criterion, optimizer=optimizer)
            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))
            if opt.new:
                if opt.version == 'v2q':
                    print('===> Quantization...')
                    quantize(model, opt.quant_bit)
                # every 5 epochs
                if (epoch+1) % opt.save_epoch == 0:
                    print('===> Change kernels using {}'.format(opt.version))
                    idx = find_similar_kernel_n_change(model, opt.version)

        # evaluate on validation set
        print('===> [ Validation ]')
        start_time = time.time()
        acc1_valid, acc5_valid = validate(val_loader, model, criterion)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print('====> {:.2f} seconds to validate this epoch\n'.format(
            elapsed_time))

        # remember best Acc@1 and save checkpoint and summary csv file
        if opt.retrain:
            if opt.new:
                # every 'opt.save_epoch' epochs
                if (epoch+1) % opt.save_epoch == 0:
                    is_best = acc1_valid > best_acc1
                    best_acc1 = max(acc1_valid, best_acc1)
                    state = {'epoch': epoch + 1,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'n_retrain': n_retrain,
                            'new': True,
                            'version': version,
                            'idx': idx}
                    summary = [epoch,
                            str(acc1_train)[7:-18], str(acc5_train)[7:-18],
                            str(acc1_valid)[7:-18], str(acc5_valid)[7:-18]]
                    save_model(state, epoch, is_best, opt, n_retrain)
                    save_summary(summary, opt, n_retrain)
            else:
                is_best = acc1_valid > best_acc1
                best_acc1 = max(acc1_valid, best_acc1)
                state = {'epoch': epoch + 1,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'n_retrain': n_retrain,
                         'new': False}
                summary = [epoch,
                           str(acc1_train)[7:-18], str(acc5_train)[7:-18],
                           str(acc1_valid)[7:-18], str(acc5_valid)[7:-18]]
                save_model(state, epoch, is_best, opt, n_retrain)
                save_summary(summary, opt, n_retrain)
        else:
            if not opt.new:
                is_best = acc1_valid > best_acc1
                best_acc1 = max(acc1_valid, best_acc1)
                state = {'epoch': epoch + 1,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'n_retrain': n_retrain,
                         'new': False}
                summary = [epoch,
                           str(acc1_train)[7:-18], str(acc5_train)[7:-18],
                           str(acc1_valid)[7:-18], str(acc5_valid)[7:-18]]
                save_model(state, epoch, is_best, opt, n_retrain)
                save_summary(summary, opt, n_retrain)
            else:
                # every 'opt.save_epoch' epochs
                if (epoch+1) % opt.save_epoch == 0:
                    is_best = acc1_valid > best_acc1
                    best_acc1 = max(acc1_valid, best_acc1)
                    state = {'epoch': epoch + 1,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'n_retrain': n_retrain,
                             'new': True,
                             'version': opt.version,
                             'idx': idx}
                    summary = [epoch,
                               str(acc1_train)[7:-18], str(acc5_train)[7:-18],
                               str(acc1_valid)[7:-18], str(acc5_valid)[7:-18]]
                    save_model(state, epoch, is_best, opt, n_retrain)
                    save_summary(summary, opt, n_retrain)

    avg_train_time = train_time / (opt.epochs-start_epoch)
    avg_valid_time = validate_time / (opt.epochs-start_epoch)
    total_train_time = train_time + validate_time
    print('====> average training time per epoch: {:,}m {:.2f}s'.format(
        int(avg_train_time//60), avg_train_time%60))
    print('====> average validation time per epoch: {:,}m {:.2f}s'.format(
        int(avg_valid_time//60), avg_valid_time%60))
    print('====> training time: {}h {}m {:.2f}s'.format(
        int(train_time//3600), int((train_time%3600)//60), train_time%60))
    print('====> validation time: {}h {}m {:.2f}s'.format(
        int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
    print('====> total training time: {}h {}m {:.2f}s'.format(
        int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))


def train(train_loader, **kwargs):
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
        if opt.nuc_loss:
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
            for idx in range(0, first_conv.size()[0], d):
                sub_tensor = torch.unsqueeze(torch.flatten(first_conv[idx:idx+d]), 0)
                if idx == 0:
                    first_conv_all = sub_tensor
                else:
                    first_conv_all = torch.cat((first_conv_all, sub_tensor), 0)
            inv_nuc_norm_regularity = opt.nls / torch.norm(first_conv_all, p='nuc')
            loss = criterion(output, target) + inv_nuc_norm_regularity
        elif opt.pcc_loss:
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
            sub_tensors = []
            for idx in range(0, first_conv.size()[0], d):
                sub_tensors.append(torch.flatten(first_conv[idx:idx+d]))
            sum_sqpcc = 0.0
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
                    sum_sqpcc += pcc_xy*pcc_xy
            inv_sqpcc_regularity = opt.pls / sum_sqpcc
            loss = criterion(output, target) + inv_sqpcc_regularity
        else:
            loss = criterion(output, target)

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

    return top1.avg, top5.avg


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


def find_similar_kernel_n_change(model, version):
    """ find the most similar kernel
    """
    if opt.arch in hasDiffLayersArchs:
        try:
            w_conv = model.module.get_weights_conv(use_cuda=True)
        except:
            if opt.cuda:
                w_conv = model.get_weights_conv(use_cuda=True)
            else:
                w_conv = model.get_weights_conv(use_cuda=False)
    else:
        try:
            w_dwconv = model.module.get_weights_dwconv(use_cuda=True)
        except:
            if opt.cuda:
                w_dwconv = model.get_weights_dwconv(use_cuda=True)
            else:
                w_dwconv = model.get_weights_dwconv(use_cuda=False)

    start_layer = 1
    ref_layer = 0
    if version in ['v1', 'v2a', 'v3a']:
        start_layer = 2
        ref_layer = 1

    idx_all = []
    if opt.arch in hasDiffLayersArchs:
        if version == 'v3' or version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_conv[ref_layer])):
                for k in range(len(w_conv[ref_layer][j])):
                    concat_kernels_ref.append(w_conv[ref_layer][j][k])
            num_subvec_ref = len(concat_kernels_ref) // d
            for i in tqdm(range(start_layer, len(w_conv)), ncols=80, unit='layer'):
                idx = []
                num_subvec_cur = (len(w_conv[i])*len(w_conv[i][0])) // d
                num_subvec_cur_each_kernel = len(w_conv[i][0]) // d
                for j in range(num_subvec_cur):
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
            for i in tqdm(range(start_layer, len(w_conv)), ncols=80, unit='layer'):
                idx = []
                for j in range(len(w_conv[i])):
                    for k in range(len(w_conv[i][j])):
                        min_diff = math.inf
                        ref_idx = 0
                        if version == 'v1':
                            for v in range(len(w_conv[ref_layer])):
                                for w in range(len(w_conv[ref_layer][v])):
                                    diff = np.sum(np.absolute(w_conv[ref_layer][v][w] - w_conv[i][j][k]))
                                    if min_diff > diff:
                                        min_diff = diff
                                        ref_idx = v * len(w_conv[ref_layer]) + w
                        elif version in ['v2', 'v2a', 'v2q']:
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
        if version == 'v3' or version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_dwconv[ref_layer])):
                for k in range(len(w_dwconv[ref_layer][j])):
                    concat_kernels_ref.append(w_dwconv[ref_layer][j][k])
            num_subvec_ref = len(concat_kernels_ref) // d
            for i in tqdm(range(start_layer, len(w_dwconv)), ncols=80, unit='layer'):
                idx = []
                num_subvec_cur = (len(w_dwconv[i])*len(w_dwconv[i][0])) // d
                for j in range(num_subvec_cur):
                    min_diff = math.inf
                    ref_idx = 0
                    for u in range(num_subvec_ref):
                        mean_cur = np.mean(w_dwconv[i][d*j:d*j+d][0])
                        mean_ref = np.mean(concat_kernels_ref[d*u:d*(u+1)])
                        alpha_numer = 0.0
                        alpha_denom = 0.0
                        for v in range(d):
                            for row in range(len(concat_kernels_ref[d*u+v])):
                                for col in range(len(concat_kernels_ref[d*u+v][row])):
                                    alpha_numer += ((concat_kernels_ref[d*u+v][row][col] - mean_ref) *
                                                    (w_dwconv[i][d*j+v][0][row][col] - mean_cur))
                                    alpha_denom += ((concat_kernels_ref[d*u+v][row][col] - mean_ref) *
                                                    (concat_kernels_ref[d*u+v][row][col] - mean_ref))
                        alpha = alpha_numer / alpha_denom
                        beta = mean_cur - alpha*mean_ref
                        diff = 0.0
                        for v in range(d):
                            tmp_diff = alpha*concat_kernels_ref[d*u+v]+beta - w_dwconv[i][d*j+v][0]
                            diff += np.sum(np.absolute(tmp_diff))
                        if min_diff > diff:
                            min_diff = diff
                            ref_idx = (u, alpha, beta)
                    idx.append(ref_idx)
                idx_all.append(idx)
        else:
            for i in tqdm(range(start_layer, len(w_dwconv)), ncols=80, unit='layer'):
                idx = []
                for j in range(len(w_dwconv[i])):
                    min_diff = math.inf
                    ref_idx = 0
                    if version == 'v1':
                        for k in range(len(w_dwconv[ref_layer])):
                            diff = np.sum(np.absolute(w_dwconv[ref_layer][k][0] - w_dwconv[i][j][0]))
                            if min_diff > diff:
                                min_diff = diff
                                ref_idx = k
                    elif version in ['v2', 'v2a', 'v2q']:
                        for k in range(len(w_dwconv[ref_layer])):
                            # find alpha, beta using least squared method every kernel in reference layer
                            mean_cur = np.mean(w_dwconv[i][j][0])
                            mean_ref = np.mean(w_dwconv[ref_layer][k][0])
                            alpha_numer = 0.0
                            alpha_denom = 0.0
                            for u in range(3):
                                for v in range(3):
                                    alpha_numer += ((w_dwconv[ref_layer][k][0][u][v] - mean_ref) *
                                                    (w_dwconv[i][j][0][u][v] - mean_cur))
                                    alpha_denom += ((w_dwconv[ref_layer][k][0][u][v] - mean_ref) *
                                                    (w_dwconv[ref_layer][k][0][u][v] - mean_ref))
                            alpha = alpha_numer / alpha_denom
                            beta = mean_cur - alpha*mean_ref
                            diff = np.sum(np.absolute(alpha*w_dwconv[ref_layer][k][0]+beta - w_dwconv[i][j][0]))
                            if min_diff > diff:
                                min_diff = diff
                                ref_idx = (k, alpha, beta)
                    elif version == 'v3' or version == 'v3a':
                        pass
                    idx.append(ref_idx)
                idx_all.append(idx)

    if opt.arch in hasDiffLayersArchs:
        if version == 'v1':
            for i in range(start_layer, len(w_conv)):
                for j in range(len(w_conv[i])):
                    for k in range(len(w_conv[i][j])):
                        ref_idx = idx_all[i-start_layer][j*len(w_conv[i][j])+k]
                        v = ref_idx // len(w_conv[ref_layer])
                        w = ref_idx % len(w_conv[ref_layer])
                        w_conv[i][j][k] = w_conv[ref_layer][v][w]
        elif version in ['v2', 'v2a', 'v2q']:
            for i in range(start_layer, len(w_conv)):
                for j in range(len(w_conv[i])):
                    for k in range(len(w_conv[i][j])):
                        ref_idx, alpha, beta = idx_all[i-start_layer][j*len(w_conv[i][j])+k]
                        v = ref_idx // len(w_conv[ref_layer])
                        w = ref_idx % len(w_conv[ref_layer])
                        w_conv[i][j][k] = alpha * w_conv[ref_layer][v][w] + beta
        elif version == 'v3' or version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_conv[ref_layer])):
                for k in range(len(w_conv[ref_layer][j])):
                    concat_kernels_ref.append(w_conv[ref_layer][j][k])
            for i in range(start_layer, len(w_conv)):
                num_subvec_cur = (len(w_conv[i])*len(w_conv[i][0])) // d
                num_subvec_cur_each_kernel = len(w_conv[i][0]) // d
                for j in range(num_subvec_cur):
                    ref_idx, alpha, beta = idx_all[i-start_layer][j]
                    subvec_idx_j = j // num_subvec_cur_each_kernel
                    subvec_idx_k = j % num_subvec_cur_each_kernel
                    for v in range(d):
                        w_conv[i][subvec_idx_j][d*subvec_idx_k+v] =\
                            alpha * concat_kernels_ref[ref_idx+v] + beta
        else:
            print('Wrong version! program exit...')
            exit()
    else:
        if version == 'v1':
            for i in range(start_layer, len(w_dwconv)):
                for j in range(len(w_dwconv[i])):
                    k = idx_all[i-start_layer][j]
                    w_dwconv[i][j] = w_dwconv[ref_layer][k]
        elif version in ['v2', 'v2a', 'v2q']:
            for i in range(start_layer, len(w_dwconv)):
                for j in range(len(w_dwconv[i])):
                    k, alpha, beta = idx_all[i-start_layer][j]
                    w_dwconv[i][j] = alpha * w_dwconv[ref_layer][k] + beta
        elif version == 'v3' or version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_dwconv[ref_layer])):
                for k in range(len(w_dwconv[ref_layer][j])):
                    concat_kernels_ref.append(w_dwconv[ref_layer][j][k])
            for i in range(start_layer, len(w_dwconv)):
                num_subvec_cur = (len(w_dwconv[i])*len(w_dwconv[i][0])) // d
                for j in range(num_subvec_cur):
                    ref_idx, alpha, beta = idx_all[i-start_layer][j]
                    for v in range(d):
                        w_dwconv[i][d*j+v][0] =\
                            alpha * concat_kernels_ref[ref_idx+v] + beta
        else:
            print('Wrong version! program exit...')
            exit()

    if opt.arch in hasDiffLayersArchs:
        try:
            model.module.set_weights_conv(w_conv, use_cuda=True)
        except:
            if opt.cuda:
                model.set_weights_conv(w_conv, use_cuda=True)
            else:
                model.set_weights_conv(w_conv, use_cuda=False)
    else:
        try:
            model.module.set_weights_dwconv(w_dwconv, use_cuda=True)
        except:
            if opt.cuda:
                model.set_weights_dwconv(w_dwconv, use_cuda=True)
            else:
                model.set_weights_dwconv(w_dwconv, use_cuda=False)
    
    return idx_all


def idxtoweight(model, indices, version):
    """change indices to weights
    """
    if opt.arch in hasDiffLayersArchs:
        try:
            w_conv = model.module.get_weights_conv(use_cuda=True)
        except:
            if opt.cuda:
                w_conv = model.get_weights_conv(use_cuda=True)
            else:
                w_conv = model.get_weights_conv(use_cuda=False)
    else:
        try:
            w_dwconv = model.module.get_weights_dwconv(use_cuda=True)
        except:
            if opt.cuda:
                w_dwconv = model.get_weights_dwconv(use_cuda=True)
            else:
                w_dwconv = model.get_weights_dwconv(use_cuda=False)

    start_layer = 1
    ref_layer = 0
    if version in ['v1', 'v2a', 'v3a']:
        start_layer = 2
        ref_layer = 1

    if opt.arch in hasDiffLayersArchs:
        if version == 'v1':
            for i in range(start_layer, len(w_conv)):
                for j in range(len(w_conv[i])):
                    for k in range(len(w_conv[i][j])):
                        ref_idx = indices[i-start_layer][j*len(w_conv[i][j])+k]
                        v = ref_idx // len(w_conv[ref_layer])
                        w = ref_idx % len(w_conv[ref_layer])
                        w_conv[i][j][k] = w_conv[ref_layer][v][w]
        elif version in ['v2', 'v2a', 'v2q']:
            for i in range(start_layer, len(w_conv)):
                for j in range(len(w_conv[i])):
                    for k in range(len(w_conv[i][j])):
                        ref_idx, alpha, beta = indices[i-start_layer][j*len(w_conv[i][j])+k]
                        v = ref_idx // len(w_conv[ref_layer])
                        w = ref_idx % len(w_conv[ref_layer])
                        w_conv[i][j][k] = alpha * w_conv[ref_layer][v][w] + beta
        elif version == 'v3' or version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_conv[ref_layer])):
                for k in range(len(w_conv[ref_layer][j])):
                    concat_kernels_ref.append(w_conv[ref_layer][j][k])
            for i in range(start_layer, len(w_conv)):
                num_subvec_cur = (len(w_conv[i])*len(w_conv[i][0])) // d
                num_subvec_cur_each_kernel = len(w_conv[i][0]) // d
                for j in range(num_subvec_cur):
                    ref_idx, alpha, beta = indices[i-start_layer][j]
                    subvec_idx_j = j // num_subvec_cur_each_kernel
                    subvec_idx_k = j % num_subvec_cur_each_kernel
                    for v in range(d):
                        w_conv[i][subvec_idx_j][d*subvec_idx_k+v] =\
                            alpha * concat_kernels_ref[ref_idx+v] + beta
        else:
            print('Wrong version! program exit...')
            exit()
    else:
        if version == 'v1':
            for i in range(start_layer, len(w_dwconv)):
                for j in range(len(w_dwconv[i])):
                    k = indices[i-start_layer][j]
                    w_dwconv[i][j] = w_dwconv[ref_layer][k]
        elif version in ['v2', 'v2a', 'v2q']:
            for i in range(start_layer, len(w_dwconv)):
                for j in range(len(w_dwconv[i])):
                    k, alpha, beta = indices[i-start_layer][j]
                    w_dwconv[i][j] = alpha * w_dwconv[ref_layer][k] + beta
        elif version == 'v3' or version == 'v3a':
            d = opt.bind_size
            concat_kernels_ref = []
            for j in range(len(w_dwconv[ref_layer])):
                for k in range(len(w_dwconv[ref_layer][j])):
                    concat_kernels_ref.append(w_dwconv[ref_layer][j][k])
            for i in range(start_layer, len(w_dwconv)):
                num_subvec_cur = (len(w_dwconv[i])*len(w_dwconv[i][0])) // d
                for j in range(num_subvec_cur):
                    ref_idx, alpha, beta = indices[i-start_layer][j]
                    for v in range(d):
                        w_dwconv[i][d*j+v][0] =\
                            alpha * concat_kernels_ref[ref_idx+v] + beta
        else:
            print('Wrong version! program exit...')
            exit()

    if opt.arch in hasDiffLayersArchs:
        try:
            model.module.set_weights_conv(w_conv, use_cuda=True)
        except:
            if opt.cuda:
                model.set_weights_conv(w_conv, use_cuda=True)
            else:
                model.set_weights_conv(w_conv, use_cuda=False)
    else:
        try:
            model.module.set_weights_dwconv(w_dwconv, use_cuda=True)
        except:
            if opt.cuda:
                model.set_weights_dwconv(w_dwconv, use_cuda=True)
            else:
                model.set_weights_dwconv(w_dwconv, use_cuda=False)


def quantize(model, num_bits=8):
    """quantize weights of convolution kernels
    """
    if opt.arch in hasDiffLayersArchs:
        try:
            w_conv = model.module.get_weights_conv(use_cuda=True)
        except:
            if opt.cuda:
                w_conv = model.get_weights_conv(use_cuda=True)
            else:
                w_conv = model.get_weights_conv(use_cuda=False)
    else:
        try:
            w_conv = model.module.get_weights_dwconv(use_cuda=True)
        except:
            if opt.cuda:
                w_conv = model.get_weights_dwconv(use_cuda=True)
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
        try:
            model.module.set_weights_conv(w_conv, use_cuda=True)
        except:
            if opt.cuda:
                model.set_weights_conv(w_conv, use_cuda=True)
            else:
                model.set_weights_conv(w_conv, use_cuda=False)
    else:
        try:
            model.module.set_weights_dwconv(w_conv, use_cuda=True)
        except:
            if opt.cuda:
                model.set_weights_dwconv(w_conv, use_cuda=True)
            else:
                model.set_weights_dwconv(w_conv, use_cuda=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))

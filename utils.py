import torch

import csv
import shutil
import pathlib
from copy import deepcopy
from os import remove
from os.path import isfile
from collections import OrderedDict

import models


def load_model(model, ckpt_file, main_gpu, use_cuda=True):
    if use_cuda:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(main_gpu))
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.module.load_state_dict(checkpoint['model'])
    else:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k[:]
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

    return checkpoint


def save_model(state, epoch, is_best, opt, n_retrain):
    dir_ckpt = pathlib.Path('checkpoint')
    if opt.arch in ['vgg', 'resnet', 'resnext', 'wideresnet']:
        dir_path = dir_ckpt / (opt.arch + str(opt.layers)) / opt.dataset
    else:
        dir_path = dir_ckpt / opt.arch / opt.dataset
    dir_path.mkdir(parents=True, exist_ok=True)

    file_name = 'ckpt'
    if not opt.retrain:
        if opt.new:
            file_name += '_new_{}'.format(opt.version)
            if opt.version == 'v3' or opt.version == 'v3a':
                if opt.nuc_loss:
                    file_name += '_nl{}_s{}'.format(
                        opt.nls, opt.save_epoch)
                elif opt.pcc_loss:
                    file_name += '_pl{}_s{}'.format(
                        opt.pls, opt.save_epoch)
                file_name += '_d{}'.format(opt.bind_size)
            elif opt.version in ['v2q', 'v2qq', 'v2qpq', 'v2qqpq']:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
                file_name += '_q{}'.format(opt.quant_bit)
                if opt.version in ['v2qq', 'v2qqpq']:
                    file_name += '{}{}'.format(
                        opt.quant_bit_a, opt.quant_bit_b)
                if opt.ifl:
                    file_name += '_ifl'
            else:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
    else:
        file_name += '_rt{}'.format(n_retrain)
        if opt.new:
            file_name += '_{}'.format(opt.version)
            if opt.version == 'v3' or opt.version == 'v3a':
                if opt.nuc_loss:
                    file_name += '_nl{}_s{}'.format(
                        opt.nls, opt.save_epoch)
                elif opt.pcc_loss:
                    file_name += '_pl{}_s{}'.format(
                        opt.pls, opt.save_epoch)
                file_name += '_d{}'.format(opt.bind_size)
            elif opt.version in ['v2q', 'v2qq', 'v2qpq', 'v2qqpq']:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
                file_name += '_q{}'.format(opt.quant_bit)
                if opt.version in ['v2qq', 'v2qqpq']:
                    file_name += '{}{}'.format(
                        opt.quant_bit_a, opt.quant_bit_b)
                if opt.ifl:
                    file_name += '_ifl'
            else:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
        else:
            if opt.quant:
                file_name += '_q{}'.format(opt.quant_bit)
                if opt.pq:
                    file_name += '_pq'
                if opt.ifl:
                    file_name += '_ifl'

    file_name_best = deepcopy(file_name) + '_best.pth' # baseline: ckpt_best.pth
    file_name += '_epoch_{}.pth'.format(epoch) # baseline: ckpt_epoch_{}.pth
    model_file = dir_path / file_name
    torch.save(state, model_file)

    if is_best:
        shutil.copyfile(model_file, dir_path / file_name_best)


def save_summary(summary, opt, n_retrain):
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)

    if opt.arch in ['vgg', 'resnet', 'resnext', 'wideresnet']:
        arch_name = opt.arch+str(opt.layers)
    else:
        arch_name = opt.arch

    file_name = '{}_{}'.format(arch_name, opt.dataset)
    if opt.retrain:
        file_name += '_rt{}'.format(n_retrain)
        if opt.new:
            file_name += '_{}'.format(opt.version)
            if opt.version == 'v3' or opt.version == 'v3a':
                if opt.nuc_loss:
                    file_name += '_nl{}_s{}'.format(
                        opt.nls, opt.save_epoch)
                elif opt.pcc_loss:
                    file_name += '_pl{}_s{}'.format(
                        opt.pls, opt.save_epoch)
                file_name += '_d{}'.format(opt.bind_size)
            elif opt.version in ['v2q', 'v2qq', 'v2qpq', 'v2qqpq']:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                if opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
                file_name += '_q{}'.format(opt.quant_bit)
                if opt.version in ['v2qq', 'v2qqpq']:
                    file_name += '{}{}'.format(
                        opt.quant_bit_a, opt.quant_bit_b)
                if opt.ifl:
                    file_name += '_ifl'
            else:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
        else:
            if opt.quant:
                file_name += '_q{}'.format(opt.quant_bit)
                if opt.pq:
                    file_name += '_pq'
                if opt.ifl:
                    file_name += '_ifl'
    else:
        if opt.new:
            file_name += '_new_{}'.format(opt.version)
            if opt.version == 'v3' or opt.version == 'v3a':
                if opt.nuc_loss:
                    file_name += '_nl{}_s{}'.format(
                        opt.nls, opt.save_epoch)
                elif opt.pcc_loss:
                    file_name += '_pl{}_s{}'.format(
                        opt.pls, opt.save_epoch)
                file_name += '_d{}'.format(opt.bind_size)
            elif opt.version in ['v2q', 'v2qq', 'v2qpq', 'v2qqpq']:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
                file_name += '_q{}'.format(opt.quant_bit)
                if opt.version in ['v2qq', 'v2qqpq']:
                    file_name += '{}{}'.format(
                        opt.quant_bit_a, opt.quant_bit_b)
                if opt.ifl:
                    file_name += '_ifl'
            else:
                if opt.nuc_loss:
                    file_name += '_nl{}'.format(opt.nls)
                elif opt.pcc_loss:
                    file_name += '_pl{}'.format(opt.pls)
    file_name += '.csv'
    file_summ = dir_path / file_name

    first_save_epoch = 0
    if opt.new:
        first_save_epoch = opt.save_epoch - 1

    if summary[0] == first_save_epoch:
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['Epoch', 'Acc@1_train', 'Acc@5_train', 'Acc@1_valid', 'Acc@5_valid']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = dir_path / 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


def save_eval(summary):
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)

    file_summ = dir_path / 'eval.csv'
    if not isfile(file_summ):
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['ckpt', 'Acc@1', 'Acc@5']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, opt_lr):
    """Sets the learning rate, decayed rate of 0.98 every epoch"""
    lr = opt_lr * (0.98**epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cal_num_weights_in_dwconv(weight):
    num = 0
    for i in range(len(weight)):
        num += len(weight[i])
    return num

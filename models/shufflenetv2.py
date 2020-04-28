"""ShuffleNetV2 in PyTorch.
See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
(https://arxiv.org/abs/1807.11164)
for more details.
"""
import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    # get depth-wise convolutional layer
    def get_layer_dwconv(self, layer_num=0):
        if layer_num < (len(self.stage2)+1):
            if layer_num == 0:
                return self.stage2[layer_num].branch1[0]
            else:
                return self.stage2[layer_num-1].branch2[3]
        elif layer_num < (len(self.stage2)+len(self.stage3)+2):
            layer_num = layer_num - (len(self.stage2)+1)
            if layer_num == 0:
                return self.stage3[layer_num].branch1[0]
            else:
                return self.stage3[layer_num-1].branch2[3]
        elif layer_num < self.get_num_dwconv_layer():
            layer_num = layer_num - (len(self.stage2)+len(self.stage3)+2)
            if layer_num == 0:
                return self.stage4[layer_num].branch1[0]
            else:
                return self.stage4[layer_num-1].branch2[3]
        else:
            print('Wrong layer number!!')
            exit()

    # get weights of dwconv
    def get_weights_dwconv(self, use_cuda=True):
        w_dwconv = []
        for i in range(len(self.stage2)):
            if use_cuda:
                if i == 0:
                    _w_dwconv1 = self.stage2[i].branch1[0].weight.cpu().detach().numpy()
                    _w_dwconv2 = self.stage2[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage2[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv)
            else:
                if i == 0:
                    _w_dwconv1 = self.stage2[i].branch1[0].weight.detach().numpy()
                    _w_dwconv2 = self.stage2[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage2[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv)
        for i in range(len(self.stage3)):
            if use_cuda:
                if i == 0:
                    _w_dwconv1 = self.stage3[i].branch1[0].weight.cpu().detach().numpy()
                    _w_dwconv2 = self.stage3[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage3[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv)
            else:
                if i == 0:
                    _w_dwconv1 = self.stage3[i].branch1[0].weight.detach().numpy()
                    _w_dwconv2 = self.stage3[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage3[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv)
        for i in range(len(self.stage4)):
            if use_cuda:
                if i == 0:
                    _w_dwconv1 = self.stage4[i].branch1[0].weight.cpu().detach().numpy()
                    _w_dwconv2 = self.stage4[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage4[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv)
            else:
                if i == 0:
                    _w_dwconv1 = self.stage4[i].branch1[0].weight.detach().numpy()
                    _w_dwconv2 = self.stage4[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage4[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv)
        return w_dwconv

    # set weights of dwconv
    def set_weights_dwconv(self, weight, use_cuda=True):
        if use_cuda:
            gpuid = self.stage2[0].branch1[0].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        for i in range(len(self.stage2)+1):
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[i]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[i]).float()
            if i == 0:
                self.stage2[i].branch1[0].weight.data.copy_(weight_tensor)
            else:
                self.stage2[i-1].branch2[3].weight.data.copy_(weight_tensor)
        for i in range(len(self.stage3)+1):
            k = i + len(self.stage2) + 1
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[k]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[k]).float()
            if i == 0:
                self.stage3[i].branch1[0].weight.data.copy_(weight_tensor)
            else:
                self.stage3[i-1].branch2[3].weight.data.copy_(weight_tensor)
        for i in range(len(self.stage4)+1):
            k = i + len(self.stage2) + len(self.stage3) + 2
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[k]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[k]).float()
            if i == 0:
                self.stage4[i].branch1[0].weight.data.copy_(weight_tensor)
            else:
                self.stage4[i-1].branch2[3].weight.data.copy_(weight_tensor)

    # get total number of dwconv layer
    def get_num_dwconv_layer(self):
        return len(self.stage2)+len(self.stage3)+len(self.stage4)+3


class ShuffleNetV2_CIFAR(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=10):
        super(ShuffleNetV2_CIFAR, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        self._stage_repeats = stages_repeats

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    # get depth-wise convolutional layer
    def get_layer_dwconv(self, layer_num=0):
        if layer_num < (len(self.stage2)+1):
            if layer_num == 0:
                return self.stage2[layer_num].branch1[0]
            else:
                return self.stage2[layer_num-1].branch2[3]
        elif layer_num < (len(self.stage2)+len(self.stage3)+2):
            layer_num = num_layer - (len(self.stage2)+1)
            if layer_num == 0:
                return self.stage3[layer_num].branch1[0]
            else:
                return self.stage3[layer_num-1].branch2[3]
        elif layer_num < self.get_num_dwconv_layer():
            layer_num = num_layer - (len(self.stage2)+len(self.stage3)+2)
            if layer_num == 0:
                return self.stage4[layer_num].branch1[0]
            else:
                return self.stage4[layer_num-1].branch2[3]
        else:
            print('Wrong layer number!!')
            exit()

    # get weights of dwconv
    def get_weights_dwconv(self, use_cuda=True):
        w_dwconv = []
        for i in range(len(self.stage2)):
            if use_cuda:
                if i == 0:
                    _w_dwconv1 = self.stage2[i].branch1[0].weight.cpu().detach().numpy()
                    _w_dwconv2 = self.stage2[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage2[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv)
            else:
                if i == 0:
                    _w_dwconv1 = self.stage2[i].branch1[0].weight.detach().numpy()
                    _w_dwconv2 = self.stage2[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage2[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv)
        for i in range(len(self.stage3)):
            if use_cuda:
                if i == 0:
                    _w_dwconv1 = self.stage3[i].branch1[0].weight.cpu().detach().numpy()
                    _w_dwconv2 = self.stage3[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage3[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv)
            else:
                if i == 0:
                    _w_dwconv1 = self.stage3[i].branch1[0].weight.detach().numpy()
                    _w_dwconv2 = self.stage3[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage3[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv)
        for i in range(len(self.stage4)):
            if use_cuda:
                if i == 0:
                    _w_dwconv1 = self.stage4[i].branch1[0].weight.cpu().detach().numpy()
                    _w_dwconv2 = self.stage4[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage4[i].branch2[3].weight.cpu().detach().numpy()
                    w_dwconv.append(_w_dwconv)
            else:
                if i == 0:
                    _w_dwconv1 = self.stage4[i].branch1[0].weight.detach().numpy()
                    _w_dwconv2 = self.stage4[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv1)
                    w_dwconv.append(_w_dwconv2)
                else:
                    _w_dwconv = self.stage4[i].branch2[3].weight.detach().numpy()
                    w_dwconv.append(_w_dwconv)
        return w_dwconv

    # set weights of dwconv
    def set_weights_dwconv(self, weight, use_cuda=True):
        if use_cuda:
            gpuid = self.stage2[0].branch1[0].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        for i in range(len(self.stage2)+1):
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[i]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[i]).float()
            if i == 0:
                self.stage2[i].branch1[0].weight.data.copy_(weight_tensor)
            else:
                self.stage2[i-1].branch2[3].weight.data.copy_(weight_tensor)
        for i in range(len(self.stage3)+1):
            k = i + len(self.stage2) + 1
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[k]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[k]).float()
            if i == 0:
                self.stage3[i].branch1[0].weight.data.copy_(weight_tensor)
            else:
                self.stage3[i-1].branch2[3].weight.data.copy_(weight_tensor)
        for i in range(len(self.stage4)+1):
            k = i + len(self.stage2) + len(self.stage3) + 2
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[k]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[k]).float()
            if i == 0:
                self.stage4[i].branch1[0].weight.data.copy_(weight_tensor)
            else:
                self.stage4[i-1].branch2[3].weight.data.copy_(weight_tensor)

    # get total number of dwconv layer
    def get_num_dwconv_layer(self):
        return len(self.stage2)+len(self.stage3)+len(self.stage4)+3


def shufflenetv2(data='cifar10', **kwargs):
    width_mult = kwargs.get('width_mult')
    out_channels = {
        0.5:[24, 48, 96, 192, 1024],
        1.0:[24, 116, 232, 464, 1024],
        1.5:[24, 176, 352, 704, 1024],
        2.0:[24, 244, 488, 976, 2048]
    }
    if data in ['cifar10', 'cifar100']:
        return ShuffleNetV2_CIFAR([4, 8, 4], out_channels[width_mult], int(data[5:]))
    elif data == 'imagenet':
        return ShuffleNetV2([4, 8, 4], out_channels[width_mult], 1000)
    # TODO:
    # elif data == 'tinyimagenet':
    #     return ShuffleNetV2([4, 8, 4], out_channels[width_mult], 100)
    else:
        return None

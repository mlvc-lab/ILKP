"""ResNet/WideResNet in PyTorch.
See the paper "Deep Residual Learning for Image Recognition"
(https://arxiv.org/abs/1512.03385)
and the paper "Wide Residual Networks"
(https://arxiv.org/abs/1605.07146)
for more details.
"""
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicDropoutBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, drop_rate=0.3):
        super(BasicDropoutBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, drop_rate=0.0):
        super(ResNet, self).__init__()
        self.block_name = str(block.__name__)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.drop_rate = drop_rate

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: 
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if self.block_name == 'BasicDropoutBlock':
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, self.drop_rate))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if self.block_name == 'BasicDropoutBlock':
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, drop_rate=self.drop_rate))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    #TODO: Bottleneck block?????? ??? ??? ?????? ????????????.. (ResNet ?????????, wideresnet ???..)
    # get convolutional layer
    def get_layer_conv(self, layer_num=0):
        if layer_num < (2*len(self.layer1)):
            if layer_num % 2 == 0:
                return self.layer1[layer_num//2].conv1
            else:
                return self.layer1[layer_num//2].conv2
        elif layer_num < (2*len(self.layer1) + 2*len(self.layer2)):
            layer_num = layer_num - 2*len(self.layer1)
            if layer_num % 2 == 0:
                return self.layer2[layer_num//2].conv1
            else:
                return self.layer2[layer_num//2].conv2
        elif layer_num < (2*len(self.layer1) + 2*len(self.layer2) + 2*len(self.layer3)):
            layer_num = layer_num - 2*len(self.layer1) - 2*len(self.layer2)
            if layer_num % 2 == 0:
                return self.layer3[layer_num//2].conv1
            else:
                return self.layer3[layer_num//2].conv2
        elif layer_num < self.get_num_conv_layer():
            layer_num = layer_num - 2*len(self.layer1) - 2*len(self.layer2) - 2*len(self.layer3)
            if layer_num % 2 == 0:
                return self.layer4[layer_num//2].conv1
            else:
                return self.layer4[layer_num//2].conv2
        else:
            print('Wrong layer number!!')
            exit()

    # get weights of convolutional layers
    def get_weights_conv(self, use_cuda=True):
        w_conv = []
        for i in range(len(self.layer1)):
            if use_cuda:
                w_conv.append(self.layer1[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer1[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer1[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer1[i].conv2.weight.detach().numpy())
        for i in range(len(self.layer2)):
            if use_cuda:
                w_conv.append(self.layer2[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer2[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer2[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer2[i].conv2.weight.detach().numpy())
        for i in range(len(self.layer3)):
            if use_cuda:
                w_conv.append(self.layer3[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer3[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer3[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer3[i].conv2.weight.detach().numpy())
        for i in range(len(self.layer4)):
            if use_cuda:
                w_conv.append(self.layer4[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer4[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer4[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer4[i].conv2.weight.detach().numpy())
        return w_conv

    def set_weights_conv(self, weight, use_cuda=True):
        if use_cuda:
            gpuid = self.conv1.weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        # residual block
        for i in range(len(self.layer1)):
            k = 2*i
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer1[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer1[i].conv2.weight.data.copy_(weight_tensor_conv2)
        for i in range(len(self.layer2)):
            k = 2*i + 2*len(self.layer1)
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer2[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer2[i].conv2.weight.data.copy_(weight_tensor_conv2)
        for i in range(len(self.layer3)):
            k = 2*i + 2*len(self.layer1) + 2*len(self.layer2)
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer3[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer3[i].conv2.weight.data.copy_(weight_tensor_conv2)
        for i in range(len(self.layer4)):
            k = 2*i + 2*len(self.layer1) + 2*len(self.layer2) + 2*len(self.layer3)
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer4[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer4[i].conv2.weight.data.copy_(weight_tensor_conv2)

    def get_num_conv_layer(self):
        return 2*len(self.layer1) + 2*len(self.layer2) + 2*len(self.layer3) + 2*len(self.layer4)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, drop_rate=0.0):
        super(ResNet_CIFAR, self).__init__()
        self.block_name = str(block.__name__)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.drop_rate = drop_rate

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: 
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if self.block_name == 'BasicDropoutBlock':
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, self.drop_rate))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if self.block_name == 'BasicDropoutBlock':
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, drop_rate=self.drop_rate))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    # get convolutional layer
    def get_layer_conv(self, layer_num=0):
        if layer_num == 0:
            return self.conv1
        elif layer_num < (2*len(self.layer1) + 1):
            layer_num = layer_num - 1
            if layer_num % 2 == 0:
                return self.layer1[layer_num//2].conv1
            else:
                return self.layer1[layer_num//2].conv2
        elif layer_num < (2*len(self.layer1) + 2*len(self.layer2) + 1):
            layer_num = layer_num - 2*len(self.layer1) - 1
            if layer_num % 2 == 0:
                return self.layer2[layer_num//2].conv1
            else:
                return self.layer2[layer_num//2].conv2
        elif layer_num < self.get_num_conv_layer():
            layer_num = layer_num - 2*len(self.layer1) - 2*len(self.layer2) - 1
            if layer_num % 2 == 0:
                return self.layer3[layer_num//2].conv1
            else:
                return self.layer3[layer_num//2].conv2
        else:
            print('Wrong layer number!!')
            exit()

    # get weights of convolutional layers
    def get_weights_conv(self, use_cuda=True):
        w_conv = []
        if use_cuda:
            w_conv.append(self.conv1.weight.cpu().detach().numpy())
        else:
            w_conv.append(self.conv1.weight.detach().numpy())
        for i in range(len(self.layer1)):
            if use_cuda:
                w_conv.append(self.layer1[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer1[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer1[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer1[i].conv2.weight.detach().numpy())
        for i in range(len(self.layer2)):
            if use_cuda:
                w_conv.append(self.layer2[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer2[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer2[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer2[i].conv2.weight.detach().numpy())
        for i in range(len(self.layer3)):
            if use_cuda:
                w_conv.append(self.layer3[i].conv1.weight.cpu().detach().numpy())
                w_conv.append(self.layer3[i].conv2.weight.cpu().detach().numpy())
            else:
                w_conv.append(self.layer3[i].conv1.weight.detach().numpy())
                w_conv.append(self.layer3[i].conv2.weight.detach().numpy())
        return w_conv

    # set weights of convolutional layers
    def set_weights_conv(self, weight, use_cuda=True):
        if use_cuda:
            gpuid = self.conv1.weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        # 1st conv layer
        if use_cuda:
            weight_tensor = torch.from_numpy(weight[0]).float().to(device)
        else:
            weight_tensor = torch.from_numpy(weight[0]).float()
        self.conv1.weight.data.copy_(weight_tensor)
        # residual block
        for i in range(len(self.layer1)):
            k = 2*i + 1
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer1[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer1[i].conv2.weight.data.copy_(weight_tensor_conv2)
        for i in range(len(self.layer2)):
            k = 2*i + 1 + 2*len(self.layer1)
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer2[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer2[i].conv2.weight.data.copy_(weight_tensor_conv2)
        for i in range(len(self.layer3)):
            k = 2*i + 1 + 2*len(self.layer1) + 2*len(self.layer2)
            if use_cuda:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float().to(device)
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float().to(device)
            else:
                weight_tensor_conv1 = torch.from_numpy(weight[k]).float()
                weight_tensor_conv2 = torch.from_numpy(weight[k+1]).float()
            self.layer3[i].conv1.weight.data.copy_(weight_tensor_conv1)
            self.layer3[i].conv2.weight.data.copy_(weight_tensor_conv2)

    # get total number of convolutional layers
    def get_num_conv_layer(self):
        return 2*len(self.layer1) + 2*len(self.layer2) + 2*len(self.layer3) + 1


# Model configurations
cfgs = {
    '18':  (BasicBlock, [2, 2, 2, 2]),
    '34':  (BasicBlock, [3, 4, 6, 3]),
    '50':  (Bottleneck, [3, 4, 6, 3]),
    '101': (Bottleneck, [3, 4, 23, 3]),
    '152': (Bottleneck, [3, 8, 36, 3]),
}
cfgs_cifar = {
    '14':  [2, 2, 2],
    '20':  [3, 3, 3],
    '32':  [5, 5, 5],
    '44':  [7, 7, 7],
    '56':  [9, 9, 9],
    '110': [18, 18, 18],
    '1202': [200, 200, 200],
}
cfgs_wrn = {
    '18':  (BasicBlock, [2, 2, 2, 2]),
    '34':  (BasicBlock, [3, 4, 6, 3]),
    '50':  (Bottleneck, [3, 4, 6, 3]),
    '101': (Bottleneck, [3, 4, 23, 3]),
}
cfgs_wrn_cifar = {
    '16':  [2, 2, 2],
    '22':  [3, 3, 3],
    '28':  [4, 4, 4],
    '40':  [6, 6, 6],
    '52':  [8, 8, 8],
}


def resnet(data='cifar10', **kwargs):
    r"""ResNet models from "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)"

    Args:
        data (str): the name of datasets
    """
    num_layers = str(kwargs.get('num_layers'))
    if data in ['cifar10', 'cifar100']:
        if num_layers in cfgs_cifar.keys():
            return ResNet_CIFAR(BasicBlock, cfgs_cifar[num_layers], int(data[5:]))
        else:
            return None
    elif data == 'imagenet':
        if num_layers in cfgs.keys():
            block, layers = cfgs[num_layers]
            return ResNet(block, layers, 1000)
        else:
            return None
    # TODO:
    # elif data == 'tinyimagenet':
    #     return ResNet(100)
    else:
        return None


def wideresnet(data='cifar10', **kwargs):
    r"""WideResNet models from "[Wide Residual Networks](https://arxiv.org/abs/1605.07146)"

    Args:
        data (str): the name of datasets
    """
    num_layers = str(kwargs.get('num_layers'))
    width_mult = kwargs.get('width_mult')
    drop_rate = kwargs.get('drop_rate')
    if data in ['cifar10', 'cifar100']:
        if num_layers in cfgs_wrn_cifar.keys():
            return ResNet_CIFAR(BasicDropoutBlock, cfgs_wrn_cifar[num_layers], int(data[5:]),
                                width_per_group=64*width_mult, drop_rate=drop_rate)
        else:
            return None
    elif data == 'imagenet':
        if num_layers in cfgs_wrn.keys():
            block, layers = cfgs_wrn[num_layers]
            return ResNet(block, layers, 1000)
        else:
            return None
    # TODO:
    # elif data == 'tinyimagenet':
    #     return ResNet(100)
    else:
        return None
'''
def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
'''
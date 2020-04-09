"""MobileNetV2 in PyTorch
See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
for more details.
"""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    '''Original MobileNetV2'''
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    # get depth-wise convolutional layer
    def get_layer_dwconv(self, layer_num=0):
        if layer_num < self.get_num_dwconv_layer():
            return self.features[layer_num+1].conv[-3][0]
        else:
            print('Wrong layer number!!')
            exit()

    # get point-wise convolutional layer
    def get_layer_dwconv(self, layer_num=0):
        if layer_num < self.get_num_pwconv_layer():
            if layer_num == 0:
                return self.features[layer_num+1].conv[-2]
            else:
                num = (layer_num + 1) // 2
                if layer_num % 2 == 1:
                    return self.features[num+1].conv[0][0]
                else:
                    return self.features[num+1].conv[-2]
        else:
            print('Wrong layer number!!')
            exit()
    
    # get weights of dwconv
    def get_weights_dwconv(self, use_cuda=True):
        w_dwconv = []
        num_blocks = len(self.features[1:-1])
        for i in range(num_blocks):
            if use_cuda:
                _w_dwconv = self.features[i+1].conv[-3][0].weight.cpu()
            else:
                _w_dwconv = self.features[i+1].conv[-3][0].weight
            _w_dwconv = _w_dwconv.detach().numpy()
            w_dwconv.append(_w_dwconv)
        return w_dwconv
    
    # get weights of pwconv
    def get_weights_pwconv(self, use_cuda=True):
        w_pwconv = []
        num_blocks = len(self.features[1:-1])
        for i in range(num_blocks):
            if i == 0:
                if use_cuda:
                    _w_pwconv = self.features[i+1].conv[-2].weight.cpu()
                else:
                    _w_pwconv = self.features[i+1].conv[-2].weight
                _w_pwconv = _w_pwconv.detach().numpy()
                w_pwconv.append(_w_pwconv)
            else:
                if use_cuda:
                    _w_pwconv1 = self.features[i+1].conv[0][0].weight.cpu()
                    _w_pwconv2 = self.features[i+1].conv[-2].weight.cpu()
                else:
                    _w_pwconv1 = self.features[i+1].conv[0][0].weight
                    _w_pwconv2 = self.features[i+1].conv[-2].weight
                _w_pwconv1 = _w_pwconv1.detach().numpy()
                _w_pwconv2 = _w_pwconv2.detach().numpy()
                w_pwconv.append(_w_pwconv1)
                w_pwconv.append(_w_pwconv2)
        return w_pwconv
    
    # set weights of dwconv
    def set_weights_dwconv(self, weight, use_cuda=True):
        num_blocks = len(self.features[1:-1])

        if use_cuda:
            gpuid = self.features[1].conv[-3][0].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        for i in range(num_blocks):
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[i]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[i]).float()
            self.features[i+1].conv[-3][0].weight.data.copy_(weight_tensor)
    
    # set weights of pwconv
    def set_weights_pwconv(self, weight, use_cuda=True):
        num_blocks = len(self.features[1:-1])

        if use_cuda:
            gpuid = self.features[1].conv[-2].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        for i in range(num_blocks):
            if i == 0:
                if use_cuda:
                    weight_tensor = torch.from_numpy(weight[i]).float().to(device)
                else:
                    weight_tensor = torch.from_numpy(weight[i]).float()
                self.features[i+1].conv[-2].weight.data.copy_(weight_tensor)
            else:
                if use_cuda:
                    weight_tensor1 = torch.from_numpy(weight[2*i-1]).float().to(device)
                    weight_tensor2 = torch.from_numpy(weight[2*i]).float().to(device)
                else:
                    weight_tensor1 = torch.from_numpy(weight[2*i-1]).float()
                    weight_tensor2 = torch.from_numpy(weight[2*i]).float()
                self.features[i+1].conv[0][0].weight.data.copy_(weight_tensor1)
                self.features[i+1].conv[-2].weight.data.copy_(weight_tensor2)

    # get total number of dwconv layer
    def get_num_dwconv_layer(self):
        return len(self.features[1:-1])

    # get total number of pwconv layer
    def get_num_pwconv_layer(self):
        return 2 * len(self.features[1:-1]) - 1


class MobileNetV2_CIFAR(nn.Module):
    '''MobileNetV2 for CIFAR-10/100'''
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2_CIFAR, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    # get depth-wise convolutional layer
    def get_layer_dwconv(self, layer_num=0):
        if layer_num < self.get_num_dwconv_layer():
            return self.features[layer_num+1].conv[-3][0]
        else:
            print('Wrong layer number!!')
            exit()

    # get point-wise convolutional layer
    def get_layer_dwconv(self, layer_num=0):
        if layer_num < self.get_num_pwconv_layer():
            if layer_num == 0:
                return self.features[layer_num+1].conv[-2]
            else:
                num = (layer_num + 1) // 2
                if layer_num % 2 == 1:
                    return self.features[num+1].conv[0][0]
                else:
                    return self.features[num+1].conv[-2]
        else:
            print('Wrong layer number!!')
            exit()
    
    # get weights of dwconv
    def get_weights_dwconv(self, use_cuda=True):
        w_dwconv = []
        num_blocks = len(self.features[1:-1])
        for i in range(num_blocks):
            if use_cuda:
                _w_dwconv = self.features[i+1].conv[-3][0].weight.cpu()
            else:
                _w_dwconv = self.features[i+1].conv[-3][0].weight
            _w_dwconv = _w_dwconv.detach().numpy()
            w_dwconv.append(_w_dwconv)
        return w_dwconv
    
    # get weights of pwconv
    def get_weights_pwconv(self, use_cuda=True):
        w_pwconv = []
        num_blocks = len(self.features[1:-1])
        for i in range(num_blocks):
            if i == 0:
                if use_cuda:
                    _w_pwconv = self.features[i+1].conv[-2].weight.cpu()
                else:
                    _w_pwconv = self.features[i+1].conv[-2].weight
                _w_pwconv = _w_pwconv.detach().numpy()
                w_pwconv.append(_w_pwconv)
            else:
                if use_cuda:
                    _w_pwconv1 = self.features[i+1].conv[0][0].weight.cpu()
                    _w_pwconv2 = self.features[i+1].conv[-2].weight.cpu()
                else:
                    _w_pwconv1 = self.features[i+1].conv[0][0].weight
                    _w_pwconv2 = self.features[i+1].conv[-2].weight
                _w_pwconv1 = _w_pwconv1.detach().numpy()
                _w_pwconv2 = _w_pwconv2.detach().numpy()
                w_pwconv.append(_w_pwconv1)
                w_pwconv.append(_w_pwconv2)
        return w_pwconv
    
    # set weights of dwconv
    def set_weights_dwconv(self, weight, use_cuda=True):
        num_blocks = len(self.features[1:-1])

        if use_cuda:
            gpuid = self.features[1].conv[-3][0].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        for i in range(num_blocks):
            if use_cuda:
                weight_tensor = torch.from_numpy(weight[i]).float().to(device)
            else:
                weight_tensor = torch.from_numpy(weight[i]).float()
            self.features[i+1].conv[-3][0].weight.data.copy_(weight_tensor)
    
    # set weights of pwconv
    def set_weights_pwconv(self, weight, use_cuda=True):
        num_blocks = len(self.features[1:-1])

        if use_cuda:
            gpuid = self.features[1].conv[-2].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        for i in range(num_blocks):
            if i == 0:
                if use_cuda:
                    weight_tensor = torch.from_numpy(weight[i]).float().to(device)
                else:
                    weight_tensor = torch.from_numpy(weight[i]).float()
                self.features[i+1].conv[-2].weight.data.copy_(weight_tensor)
            else:
                if use_cuda:
                    weight_tensor1 = torch.from_numpy(weight[2*i-1]).float().to(device)
                    weight_tensor2 = torch.from_numpy(weight[2*i]).float().to(device)
                else:
                    weight_tensor1 = torch.from_numpy(weight[2*i-1]).float()
                    weight_tensor2 = torch.from_numpy(weight[2*i]).float()
                self.features[i+1].conv[0][0].weight.data.copy_(weight_tensor1)
                self.features[i+1].conv[-2].weight.data.copy_(weight_tensor2)

    # get total number of dwconv layer
    def get_num_dwconv_layer(self):
        return len(self.features[1:-1])

    # get total number of pwconv layer
    def get_num_pwconv_layer(self):
        return 2 * len(self.features[1:-1]) - 1


def mobilenetv2(data='cifar10', width_mult=1.0):
    if data == 'cifar10':
        return MobileNetV2_CIFAR(10, width_mult)
    elif data == 'cifar100':
        return MobileNetV2_CIFAR(100, width_mult)
    elif data == 'imagenet':
        return MobileNetV2(1000, width_mult)
    else:
        return None
    # elif data == 'tinyimagenet':
    #     return MobileNetV2(100, width_mult)

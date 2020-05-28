"""VGG in PyTorch
See the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition"
(https://arxiv.org/abs/1409.1556)
for more details.
"""
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # get convolutional layer
    def get_layer_conv(self, layer_num=0):
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if idx == layer_num:
                    return m
                idx = idx + 1
        print('Wrong layer number!!')
        exit()

    # get weights of convolutional layers
    def get_weights_conv(self, use_cuda=True):
        w_conv = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if use_cuda:
                    w_conv.append(m.weight.cpu().detach().numpy())
                else:
                    w_conv.append(m.weight.detach().numpy())
        return w_conv

    # set weights of convolutional layers
    def set_weights_conv(self, weight, use_cuda=True):
        if use_cuda:
            gpuid = self.features[0].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if use_cuda:
                    weight_tensor = torch.from_numpy(weight[idx]).float().to(device)
                else:
                    weight_tensor = torch.from_numpy(weight[idx]).float()
                m.weight.data.copy_(weight_tensor)
                idx = idx + 1

    # get total number of convolutional layers
    def get_num_conv_layer(self):
        num_layer = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                num_layer = num_layer + 1
        return num_layer


class VGG_CIFAR(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # get convolutional layer
    def get_layer_conv(self, layer_num=0):
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if idx == layer_num:
                    return m
                idx = idx + 1
        print('Wrong layer number!!')
        exit()

    # get weights of convolutional layers
    def get_weights_conv(self, use_cuda=True):
        w_conv = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if use_cuda:
                    w_conv.append(m.weight.cpu().detach().numpy())
                else:
                    w_conv.append(m.weight.detach().numpy())
        return w_conv

    # set weights of convolutional layers
    def set_weights_conv(self, weight, use_cuda=True):
        if use_cuda:
            gpuid = self.features[0].weight.get_device()
            cuda_gpu = 'cuda:' + str(gpuid)
            device = torch.device(cuda_gpu)

        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if use_cuda:
                    weight_tensor = torch.from_numpy(weight[idx]).float().to(device)
                else:
                    weight_tensor = torch.from_numpy(weight[idx]).float()
                m.weight.data.copy_(weight_tensor)
                idx = idx + 1

    # get total number of convolutional layers
    def get_num_conv_layer(self):
        num_layer = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                num_layer = num_layer + 1
        return num_layer


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# Model configurations
""" 11: VGG 11-layer model (configuration "A")
    13: VGG 13-layer model (configuration "B")
    16: VGG 16-layer model (configuration "D")
    19: VGG 19-layer model (configuration "E")
"""
cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(data='cifar10', **kwargs):
    r"""VGG models from "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)"

    Args:
        data: the name of datasets
    """
    num_layers = str(kwargs.get('num_layers'))
    batch_norm = kwargs.get('batch_norm')
    if data in ['cifar10', 'cifar100']:
        if num_layers in cfgs.keys():
            return VGG_CIFAR(make_layers(cfgs[num_layers], batch_norm=batch_norm), int(data[5:]))
        else:
            return None
    elif data == 'imagenet':
        if num_layers in cfgs.keys():
            return VGG(make_layers(cfgs[num_layers], batch_norm=batch_norm), 1000)
        else:
            return None
    # TODO:
    # elif data == 'tinyimagenet':
    #     return VGG(make_layers(cfgs[str(num_layers)], batch_norm=batch_norm), 100)
    else:
        return None

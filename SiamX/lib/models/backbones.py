import torch
import torch.nn as nn
import math


class Bottleneck(nn.Module):
    """
    modified from SiamRPN++
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet_plus2(nn.Module):
    """
    modified from SiamRPN++
    """

    def __init__(self, block, layers, used_layers, online=False):
        self.inplanes = 64
        super(ResNet_plus2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        self.layer3_use = True if 3 in used_layers else False
        self.layer4_use = True if 4 in used_layers else False

        if self.layer3_use:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if self.layer4_use:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            downsample=downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)

        p3 = self.layer3(p2)

        if self.layer4_use: p4 = self.layer4(p3)

        return [x_, p1, p2], p3


class ResNet50(nn.Module):
    def __init__(self, used_layers=[3]):
        super(ResNet50, self).__init__()
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3], used_layers=used_layers)

    def forward(self, x):
        x_stages, x = self.features(x)
        return x_stages, x


if __name__ == '__main__':
    import torch
    import numpy as np
    import time

    net = ResNet50().cuda()
    # print(net)

    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total params" + str(k / 1e6) + "M")

    search = torch.rand(1, 3, 255, 255)
    search = torch.Tensor(search).cuda()
    start = time.time()
    out = net(search)
    print(time.time() - start)
    print(len(out))
    print(len(out[0]))
    print(out[0][0].shape)
    print(out[0][1].shape)
    print(out[0][2].shape)
    print(out[1].shape)

    # print(out.size())

    print()

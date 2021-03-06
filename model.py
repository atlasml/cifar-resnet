import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn

from torchvision.models import ResNet


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


def basic_block_factory(layer_block=None):
    # Factory for using torchvision ResNet class
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     reduction=16):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

            if layer_block is not None:
                self.layer_block = layer_block(planes, reduction)
            else:
                self.layer_block = None

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.layer_block is not None:
                out = self.layer_block(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    return BasicBlock


def bottleneck_factory(layer_block=None):
    # Factory for using torchvision ResNet class
    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     reduction=16):
            super(Bottleneck, self).__init__()
            self.conv1 = conv1x1(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes, stride=stride)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = conv1x1(planes, planes * self.expansion)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

            if layer_block is not None:
                self.layer_block = layer_block(planes * self.expansion,
                                               reduction)
            else:
                self.layer_block = None

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

            if self.layer_block is not None:
                out = self.layer_block(out)

            if self.downsample:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    return Bottleneck


class CifarResNetWithBlock(nn.Module):
    def __init__(self, n_size, num_classes=10, layer_block=None,
                 reduction=None):
        super(CifarResNetWithBlock, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.block = basic_block_factory(layer_block=layer_block)
        self.layer1 = self._make_layer(16, blocks=n_size, stride=1,
                                       reduction=reduction)
        self.layer2 = self._make_layer(32, blocks=n_size, stride=2,
                                       reduction=reduction)
        self.layer3 = self._make_layer(64, blocks=n_size, stride=2,
                                       reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, reduction):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes,
                                               stride=stride),
                                       nn.BatchNorm2d(planes))

        layers.append(self.block(self.inplanes, planes, stride, downsample,
                                 reduction))
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(self.block(self.inplanes, planes,
                                     reduction=reduction))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cifar_srm_resnet32(pretrained=False, progress=True, **kwargs):
    model = CifarResNetWithBlock(5, layer_block=SRMLayer, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('just-a-model.pth'))
    return model
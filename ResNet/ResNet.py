import sys
import os

from torchsummary import summary

sys.path.append("..")
from SeNet.SeNet import SeNet
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_channel,out_channel,stride=1,downsample=None,is_se=False,**kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(3,3),
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=(3,3),
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.is_se = is_se
        if self.is_se:
            self.se = SeNet(out_channel, 16)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.is_se:
            out *= self.se(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 downsample=None,
                 is_se=False,
                 groups=1,
                 width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=width,
                               kernel_size=1,
                               stride=1,
                               bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width,
                               out_channels=width,
                               groups=groups,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width,
                               out_channels=out_channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.is_se = is_se
        if self.is_se:
            self.se = SeNet(out_channel, 16)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.is_se:
            out *= self.se(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            blocks_num,
            num_classes=1000,
            include_top=True,
            is_se=False,
            groups=1,  # 为一时表示标准的卷积，其他数时表示组卷积
            width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.is_se = is_se

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3,
                               self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          channel * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(
            block(self.in_channel,
                  channel,
                  downsample=downsample,
                  is_se=self.is_se,
                  stride=stride,
                  groups=self.groups,
                  width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(self.in_channel,
                      channel,
                      is_se=self.is_se,
                      groups=self.groups,
                      width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True, is_se=False):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, is_se=is_se)


def resnet34(num_classes=1000, include_top=True, is_se=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, is_se=is_se)


def resnet50(num_classes=1000, include_top=True, is_se=False):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, is_se=is_se)


def resnet101(num_classes=1000, include_top=True, is_se=False):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, is_se=is_se)


def resnext50_32x4d(num_classes=1000, include_top=True, is_se=False):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, is_se=is_se,
                  groups=groups, width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True, is_se=False):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, is_se=is_se,
                  groups=groups, width_per_group=width_per_group)


if __name__ == '__main__':
    #summary(resnet18(), (3, 32, 32), device="cpu")
    #summary(resnet34(), (3, 32, 32), device="cpu")
    #summary(resnet50(), (3, 32, 32), device="cpu")
    summary(resnet101(), (3, 32, 32), device="cpu")

# Copyright 2017 Queequeg92.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dual Path Networks.

Related papers:
[1] Chen, Yunpeng, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, and
    Jiashi Feng. "Dual Path Networks." arXiv preprint arXiv:1707.01629 (2017).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import math


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ConvBlock(nn.Module):
    '''Similar to PreActBlock but without shortcut connection.'''
    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)


    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class Transition(nn.Module):
    '''Transition layer between stages.
       Transition layer starts both the resnet
       branch and the densenet branch of a stage.
    '''
    def __init__(self, block, in_planes, planes, growth_rate, stride=1):
        super(Transition, self).__init__()
        self.resnet_block = block(in_planes, planes, stride)
        self.conv = conv3x3(in_planes, 2*growth_rate, stride)

    def forward(self, x):
        resnet_path = self.resnet_block(x)
        densenet_path = self.conv(x)
        out = torch.cat([resnet_path, densenet_path], 1)
        return out


class DualPathBlock(nn.Module):
    def __init__(self, block, in_planes, res_planes, growth_rate, stride=1):
        super(DualPathBlock, self).__init__()
        self.conv_block = block(in_planes, res_planes + growth_rate, stride)
        self.res_planes = res_planes

    def forward(self, x):
        merge_out = self.conv_block(x)
        out = torch.cat([x[:,:self.res_planes] + merge_out[:,:self.res_planes],   # resnet branch.
                         x[:, self.res_planes:],              # new features of densenet branch.
                         merge_out[:, self.res_planes:]], 1)  # all previous features of densenet branch.
        return out


class DualPathNet(nn.Module):
    def __init__(self, conv_block, transition_block, num_blocks, filters, growth_rates, num_classes=10):
        super(DualPathNet, self).__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(1, self.in_planes)

        self.stage1_0 = Transition(transition_block, self.in_planes, filters[0], growth_rates[0], stride=1)
        self.in_planes = filters[0] + 2 * growth_rates[0]
        self.stage1 = self._make_layer(conv_block, self.in_planes,
                                       filters[0], num_blocks[0],
                                       growth_rates[0], stride=1)

        self.stage2_0 = Transition(transition_block, self.in_planes, filters[1], growth_rates[1], stride=2)
        self.in_planes = filters[1] + 2 * growth_rates[1]
        self.stage2 = self._make_layer(conv_block, self.in_planes,
                                       filters[1], num_blocks[1],
                                       growth_rates[1], stride=1)

        self.stage3_0 = Transition(transition_block, self.in_planes, filters[2], growth_rates[2], stride=2)
        self.in_planes = filters[2] + 2 * growth_rates[2]
        self.stage3 = self._make_layer(conv_block, self.in_planes,
                                       filters[2], num_blocks[2],
                                       growth_rates[2], stride=1)

        self.bn = nn.BatchNorm2d(self.in_planes)
        self.linear = nn.Linear(self.in_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, in_planes, planes, num_blocks, growth_rate, stride):
        strides = [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(DualPathBlock(block, self.in_planes, planes, growth_rate, stride))
            self.in_planes += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.stage1_0(out)
        out = self.stage1(out)

        out = self.stage2_0(out)
        out = self.stage2(out)

        out = self.stage3_0(out)
        out = self.stage3(out)

        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def DualPathNet_28_10(num_classes=10):
    return DualPathNet(ConvBlock, PreActBlock, [4, 4, 4], [160, 320, 640], [16, 32, 64], num_classes)


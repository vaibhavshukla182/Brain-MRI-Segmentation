import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=3, dilation=2, padding=2,
                               bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        #out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate=[16,16,16,16], nDenseBlocks=[4,4,4,4], reduction=[0.7,0.7,0.7,0.7], n_classes=11, bottleneck=False):
        super(DenseNet, self).__init__()

        nChannels = 2*growthRate[0]
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        # dense 1
        self.dense1 = self._make_dense(nChannels, growthRate[0], nDenseBlocks[0], bottleneck)
        nChannels += nDenseBlocks[0]*growthRate[0]
        nOutChannels = int(math.floor(nChannels*reduction[0]))
        self.trans1 = Transition(nChannels, nOutChannels)
        # dense 2
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate[1], nDenseBlocks[1], bottleneck)
        nChannels += nDenseBlocks[1]*growthRate[1]
        nOutChannels = int(math.floor(nChannels*reduction[1]))
        self.trans2 = Transition(nChannels, nOutChannels)
        # dense 3
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate[2], nDenseBlocks[2], bottleneck)
        nChannels += nDenseBlocks[2]*growthRate[2]
        nOutChannels = int(math.floor(nChannels*reduction[2]))
        self.trans3 = Transition(nChannels, nOutChannels)
        # dense 4
        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate[3], nDenseBlocks[3], bottleneck)
        nChannels += nDenseBlocks[3]*growthRate[3]
        nOutChannels = int(math.floor(nChannels*reduction[3]))
        self.trans4 = Transition(nChannels, nOutChannels)
        self.score=nn.Sequential(
                nn.BatchNorm2d(nOutChannels),
                nn.Conv2d(nOutChannels,n_classes,1),
                #nn.Dropout(0.5),
                )

        #nChannels = nOutChannels
        #self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        #nChannels += nDenseBlocks*growthRate

        #self.bn1 = nn.BatchNorm2d(nChannels)
        #self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        pre_conv=self.conv1(x)
        dense1=self.trans1(self.dense1(pre_conv))
        dense2=self.trans2(self.dense2(dense1))
        dense3=self.trans3(self.dense3(dense2))
        dense4=self.trans4(self.dense4(dense3))
        score=self.score(dense4)
        return score

class DenseNetSeg(nn.Module):
    def __init__(self, growthRate=12, depth=16, reduction=0.5, nClasses=11, bottleneck=False):
        super(DenseNetSeg, self).__init__()
        nDenseBlocks = depth
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        self.score=nn.Sequential(
                nn.Conv2d(nChannels+growthRate*nDenseBlocks,nClasses,1),
                #nn.Dropout(0.5),
                )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
    def forward(self,x):
        pre_conv=self.conv1(x)
        dense=self.dense1(pre_conv)
        score=self.score(dense)
        return score
        



if __name__=='__main__':
    x=Variable(torch.Tensor(4,3,256,256))
    model=DenseNetSeg(growthRate=12, depth=16, reduction=0.5, nClasses=11, bottleneck=False)
    y=model(x)
    print(y.shape)

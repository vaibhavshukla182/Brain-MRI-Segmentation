import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.pool=nn.MaxPool2d(2, stride=2, ceil_mode=True)
    def forward(self,x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        conv5 = self.conv_block5(self.pool(conv4))
        return conv1,conv2,conv3,conv4,conv5
    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data


class VGG16_dilated(nn.Module):
    def __init__(self):
        super(VGG16_dilated, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.pool=nn.MaxPool2d(2, stride=2, ceil_mode=True)
    def forward(self,x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        conv5 = self.conv_block5(self.pool(conv4))
        return conv1,conv2,conv3,conv4,conv5
    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data


class VGG16_rcf(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv1_1=nn.Sequential(nn.Conv2d(3  , 64 , 3, padding=1),nn.ReLU(inplace=True),)
        self.conv1_2=nn.Sequential(nn.Conv2d(64 , 64 , 3, padding=1),nn.ReLU(inplace=True),)
        self.conv2_1=nn.Sequential(nn.Conv2d(64 , 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv2_2=nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv3_1=nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv3_2=nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv3_3=nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv4_1=nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv4_2=nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv4_3=nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv5_1=nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv5_2=nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv5_3=nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.pool=nn.MaxPool2d(2, stride=2, ceil_mode=True)
    def forward(self, x):
        conv1_1=self.conv1_1(x)
        conv1_2=self.conv1_2(conv1_1)
        conv2_1=self.conv2_1(self.pool(conv1_2))
        conv2_2=self.conv2_2(conv2_1)
        conv3_1=self.conv3_1(self.pool(conv2_2))
        conv3_2=self.conv3_2(conv3_1)
        conv3_3=self.conv3_3(conv3_2)
        conv4_1=self.conv4_1(self.pool(conv3_3))
        conv4_2=self.conv4_2(conv4_1)
        conv4_3=self.conv4_3(conv4_2)
        conv5_1=self.conv5_1(self.pool(conv4_3))
        conv5_2=self.conv5_2(conv5_1)
        conv5_3=self.conv5_3(conv5_2)
        return conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,conv3_3,conv4_1,conv4_2,conv4_3,conv5_1,conv5_2,conv5_3
    def init_vgg16_params(self,vgg16=models.vgg16(pretrained=True),copy_fc8=True):
        convs=[ self.conv1_1,self.conv1_2,
                self.conv2_1,self.conv2_2,
                self.conv3_1,self.conv3_2,self.conv3_3,
                self.conv4_1,self.conv4_2,self.conv4_3,
                self.conv5_1,self.conv5_2,self.conv5_3]
        features=list(vgg16.features.children())
        ranges=[0,2,5,7,10,12,14,17,19,21,24,26,28]
        for idx,conv in enumerate(convs):
            l1=features[ranges[idx]]
            l2=conv[0]
            if isinstance(l1,nn.Conv2d) and isinstance(l2,nn.Conv2d):
                assert l1.weight.size()==l2.weight.size()
                assert l1.bias.size()==l2.bias.size()
                l2.weight.data=l1.weight.data
                l2.bias.data=l1.bias.data



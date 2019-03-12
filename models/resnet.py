import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

#--modified from
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        C1 = x 
        x = self.layer1(x)
        C2 = x
        x = self.layer2(x)
        C3 = x 
        x = self.layer3(x)
        C4 = x
        x = self.layer4(x)
        C5 = x
        return C1, C2, C3, C4, C5


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained==True:
        state_dict=model_zoo.load_url(model_urls['resnet50'])
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict)
    return model

def resnet101(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained==True:
        state_dict=model_zoo.load_url(model_urls['resnet101'])
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict)
    return model

def resnet152(pretrained=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained==True:
        state_dict=model_zoo.load_url(model_urls['resnet152'])
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict)
    return model

class FCN_res(nn.Module):
    def __init__(self,n_classes=11,pretrained=True):
        super(FCN_res,self).__init__()
        self.n_classes=n_classes
        self.res=resnet152(pretrained=True)
        self.conv1_16=nn.Conv2d(64,  64, 3, padding=1)
        self.conv2_16=nn.Conv2d(256, 64, 3, padding=1)
        self.conv3_16=nn.Conv2d(512, 64, 3, padding=1)
        self.conv4_16=nn.Conv2d(1024, 64, 3, padding=1)
        self.conv5_16=nn.Conv2d(2048, 64, 3, padding=1)

        self.up_conv1_16 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv2_16 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv3_16 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.up_conv4_16 = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8)
        self.up_conv5_16 = nn.ConvTranspose2d(64, 64, kernel_size=16, stride=16)

        self.score=nn.Sequential(
            nn.Conv2d(5*64,self.n_classes,1),
            #nn.Dropout(0.5),
            )
    def forward(self,x):
        C1,C2,C3,C4,C5=self.res(x)

        up_conv1_16=self.up_conv1_16(self.conv1_16(C1))
        up_conv2_16=self.up_conv2_16(self.conv2_16(C2))
        up_conv3_16=self.up_conv3_16(self.conv3_16(C3))
        up_conv4_16=self.up_conv4_16(self.conv4_16(C4))
        up_conv5_16=self.up_conv5_16(self.conv5_16(C5))

        concat_1_to_5=torch.cat([up_conv1_16,up_conv2_16,up_conv3_16,up_conv4_16,up_conv5_16], 1)
        score=self.score(concat_1_to_5)
        return score
        
if __name__=='__main__':
    model=FCN_res()
    x=Variable(torch.zeros([1,3,48,48]).float())
    print(x.shape)
    y=model(x)
    print(y.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from torch.autograd import Variable
from models.resnet import resnet50

LAYER_THICK=16

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


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        #self.C5_conv = nn.Conv2d(in_channels=512, out_channels=LAYER_THICK, kernel_size=1, bias=False)
        #self.C4_conv = nn.Conv2d(in_channels=512, out_channels=LAYER_THICK, kernel_size=1, bias=False)
        #self.C3_conv = nn.Conv2d(in_channels=256, out_channels=LAYER_THICK, kernel_size=1, bias=False)
        #self.C2_conv = nn.Conv2d(in_channels=128, out_channels=LAYER_THICK, kernel_size=1, bias=False)        
        
        self.C5_conv = nn.Conv2d(in_channels=2048, out_channels=LAYER_THICK, kernel_size=1, bias=False)
        self.C4_conv = nn.Conv2d(in_channels=1024, out_channels=LAYER_THICK, kernel_size=1, bias=False)
        self.C3_conv = nn.Conv2d(in_channels=512, out_channels=LAYER_THICK, kernel_size=1, bias=False)
        self.C2_conv = nn.Conv2d(in_channels=256, out_channels=LAYER_THICK, kernel_size=1, bias=False)        

        #config = "in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False"
        #self.P2_conv = nn.Conv2d(*config)
        #self.P3_conv = nn.Conv2d(*config)
        #self.P4_conv = nn.Conv2d(*config)
        
        self.P2_conv = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=1, padding=1, bias=False)
        self.P3_conv = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=1, padding=1, bias=False)
        self.P4_conv = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=1, padding=1, bias=False)
        
         
    def forward(self, C_vector):
        C1, C2, C3, C4, C5 = C_vector
        _, _, c2_height, c2_width = C2.size()
        _, _, c3_height, c3_width = C3.size()
        _, _, c4_height, c4_width = C4.size()

        P5 = self.C5_conv(C5)

        P4 = F.upsample(P5, size=(c4_height, c4_width), mode='bilinear') + self.C4_conv(C4)
        P4 = self.P4_conv(P4)

        P3 = F.upsample(P4, size=(c3_height, c3_width), mode='bilinear') + self.C3_conv(C3)
        P3 = self.P3_conv(P3)

        P2 = F.upsample(P3, size=(c2_height, c2_width), mode='bilinear') + self.C2_conv(C2)
        P2 = self.P2_conv(P2)

        return P2, P3, P4, P5

class generateN(nn.Module):
    def __init__(self):
        super(generateN, self).__init__()
        
        config = "in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False"        
        self.N2_conv = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=2, padding=1, bias=False)
        self.N3_conv = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=2, padding=1, bias=False)
        self.N4_conv = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=2, padding=1, bias=False)

        
        config = "in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False"        
        self.N2_conv2 = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=1, padding=1, bias=False)
        self.N3_conv2 = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=1, padding=1, bias=False)
        self.N4_conv2 = nn.Conv2d(in_channels=LAYER_THICK, out_channels=LAYER_THICK, kernel_size=3, stride=1, padding=1, bias=False) 

    def forward(self, P_vec):
        [P2, P3, P4, P5] = P_vec
        N2 = P2
        
        N3 = self.N2_conv2(P3 + self.N2_conv(N2))
        N4 = self.N3_conv2(P4 + self.N3_conv(N3))
        N5 = self.N4_conv2(P5 + self.N4_conv(N4))
         
        return N2, N3, N4, N5


class PAN(nn.Module):
    def __init__(self, training=False):
        super(PAN, self).__init__()
        #self.vgg = VGG16()
        #vgg16=models.vgg16(pretrained=True)
        #self.vgg.init_vgg16_params(vgg16)
        self.res=resnet50()
        self.fpn = FPN()
        self.generateN = generateN()
        self.training = training
   
    def forward(self, x):
        """
        x : input image
        """ 
        #C1, C2, C3, C4, C5 = self.vgg(x)
        C1, C2, C3, C4, C5 = self.res(x)
        P2, P3, P4, P5 = self.fpn([C1, C2, C3, C4, C5]) 
        N2, N3, N4, N5 = self.generateN([P2, P3, P4, P5])
        return N2,N3,N4,N5

class PAN_seg(nn.Module):
    def __init__(self,n_classes=9):
        super(PAN_seg,self).__init__()
        self.n_classes=n_classes
        self.PAN=PAN()
        self.deconv2=nn.ConvTranspose2d(LAYER_THICK,LAYER_THICK,kernel_size=2,stride=2)
        self.deconv3=nn.ConvTranspose2d(LAYER_THICK,LAYER_THICK,kernel_size=4,stride=4)
        self.deconv4=nn.ConvTranspose2d(LAYER_THICK,LAYER_THICK,kernel_size=8,stride=8)
        self.deconv5=nn.ConvTranspose2d(LAYER_THICK,LAYER_THICK,kernel_size=16,stride=16)
        self.score=nn.Sequential(
                nn.Conv2d(4*LAYER_THICK,self.n_classes,1),
                nn.Dropout(0.5,)
                )
    def forward(self,x):
        conv2,conv3,conv4,conv5=self.PAN(x)
        deconv2=self.deconv2(conv2)
        deconv3=self.deconv3(conv3)
        deconv4=self.deconv4(conv4)
        deconv5=self.deconv5(conv5)
        cat=torch.cat([deconv2,deconv3,deconv4,deconv5],1)
        score=self.score(cat)
        return score

if __name__ == '__main__':
    x=torch.Tensor(4,3,16,16)
    x=Variable(x)
    print(x.shape)
    model=PAN_seg()
    y=model(x)
    print(y.shape)


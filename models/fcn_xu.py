import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class fcn_xu(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn_xu, self).__init__()
        self.n_classes = n_classes
        
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

        self.conv1_16=nn.Conv2d(64,  64, 3, padding=1)
        self.conv2_16=nn.Conv2d(128, 64, 3, padding=1)
        self.conv3_16=nn.Conv2d(256, 64, 3, padding=1)
        self.conv4_16=nn.Conv2d(512, 64, 3, padding=1)
        self.conv5_16=nn.Conv2d(512, 64, 3, padding=1)

        self.up_conv2_16 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv3_16 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.up_conv4_16 = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8)
        self.up_conv5_16 = nn.ConvTranspose2d(64, 64, kernel_size=16, stride=16)

        self.score=nn.Sequential(
            nn.Conv2d(4*64,self.n_classes,1),
            #nn.Dropout(0.5),
            )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        conv5 = self.conv_block5(self.pool(conv4))
        
        conv1_16=self.conv1_16(conv1)
        up_conv2_16=self.up_conv2_16(self.conv2_16(conv2))
        up_conv3_16=self.up_conv3_16(self.conv3_16(conv3))
        up_conv4_16=self.up_conv4_16(self.conv4_16(conv4))
        up_conv5_16=self.up_conv5_16(self.conv5_16(conv5))

        concat_1_to_5=torch.cat([up_conv2_16,up_conv3_16,up_conv4_16,up_conv5_16], 1)
        score=self.score(concat_1_to_5)
        return score

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
                

class fcn_xu_19(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn_xu_19, self).__init__()
        self.n_classes = n_classes
        
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

        self.conv1_16=nn.Conv2d(64,  64, 3, padding=1)
        self.conv2_16=nn.Conv2d(128, 64, 3, padding=1)
        self.conv3_16=nn.Conv2d(256, 64, 3, padding=1)
        self.conv4_16=nn.Conv2d(512, 64, 3, padding=1)
        self.conv5_16=nn.Conv2d(512, 64, 3, padding=1)

        self.up_conv2_16 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv3_16 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.up_conv4_16 = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8)
        self.up_conv5_16 = nn.ConvTranspose2d(64, 64, kernel_size=16, stride=16)

        self.score=nn.Sequential(
            nn.Conv2d(4*64,self.n_classes,1),
            #nn.Dropout(0.5),
            )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        conv5 = self.conv_block5(self.pool(conv4))
        
        conv1_16=self.conv1_16(conv1)
        up_conv2_16=self.up_conv2_16(self.conv2_16(conv2))
        up_conv3_16=self.up_conv3_16(self.conv3_16(conv3))
        up_conv4_16=self.up_conv4_16(self.conv4_16(conv4))
        up_conv5_16=self.up_conv5_16(self.conv5_16(conv5))

        concat_1_to_5=torch.cat([up_conv2_16,up_conv3_16,up_conv4_16,up_conv5_16], 1)
        score=self.score(concat_1_to_5)
        return score

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


class fcn_nopool(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn_nopool, self).__init__()
        self.n_classes = n_classes
        
        self.pre_conv=nn.Sequential(nn.Conv2d(3,3,1),nn.ReLU(inplace=True),)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)

        self.conv1_16=nn.Conv2d(64,  64, 3, padding=1)
        self.conv2_16=nn.Conv2d(128, 64, 3, padding=1)
        self.conv3_16=nn.Conv2d(256, 64, 3, padding=1)
        self.conv4_16=nn.Conv2d(512, 64, 3, padding=1)
        self.conv5_16=nn.Conv2d(512, 64, 3, padding=1)

        self.score=nn.Sequential(
            nn.Conv2d(4*128,self.n_classes,1),
            #nn.Dropout(0.5),
            )

    def forward(self, x):
        #x=self.pre_conv(x)
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        #conv5 = self.conv_block5(conv4)
        
        #conv1_16=self.conv1_16(conv1)
        #conv2_16=self.conv2_16(conv2)
        #conv3_16=self.conv3_16(conv3)
        #conv4_16=self.conv4_16(conv4)
        #conv5_16=self.conv5_16(conv5)

        concat_1_to_4=torch.cat([conv1,conv2,conv3,conv4], 1)
        score=self.score(concat_1_to_4)
        return score

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



class fcn_xu_dilated(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn_xu_dilated, self).__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, dilation=2, padding=2),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, dilation=2, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, dilation=3, padding=3),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=3, padding=3),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=3, padding=3),nn.ReLU(inplace=True),)

        self.conv1_16=nn.Conv2d(64, 16, 3, padding=1)
        self.conv2_16=nn.Conv2d(128, 16, 3, padding=1)
        self.conv3_16=nn.Conv2d(256, 16, 3, padding=1)
        self.conv4_16=nn.Conv2d(512, 16, 3, padding=1)
        self.conv5_16=nn.Conv2d(512, 16, 3, padding=1)

        self.score=nn.Sequential(
            nn.Conv2d(5*16,self.n_classes,1),
            nn.Dropout(0.5),
            )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        conv1_16=self.conv1_16(conv1)
        conv2_16=self.conv2_16(conv2)
        conv3_16=self.conv3_16(conv3)
        conv4_16=self.conv4_16(conv4)
        conv5_16=self.conv5_16(conv5)

        concat_1_to_4=torch.cat([conv1_16,conv2_16,conv3_16,conv4_16,conv5_16], 1)
        score=self.score(concat_1_to_4)
        return score

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


class fcn_mul(nn.Module):
    def __init__(self,in_channels=3,n_mod=3,n_feature=16,n_classes=11):
        super(fcn_mul,self).__init__()
        self.in_channels=in_channels
        self.n_mod=n_mod
        self.n_feature=n_feature
        self.n_classes=n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64 , 64 , 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64 , 128, 3, padding=1),nn.ReLU(inplace=True),
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

        self.conv1_c=nn.Conv2d(64 *self.n_mod,self.n_feature,3,padding=1)
        self.conv2_c=nn.Conv2d(128*self.n_mod,self.n_feature,3,padding=1)
        self.conv3_c=nn.Conv2d(256*self.n_mod,self.n_feature,3,padding=1)
        self.conv4_c=nn.Conv2d(512*self.n_mod,self.n_feature,3,padding=1)
        self.conv5_c=nn.Conv2d(512*self.n_mod,self.n_feature,3,padding=1)

        self.deconv2=nn.ConvTranspose2d(self.n_feature,self.n_feature,kernel_size=2 ,stride=2 )
        self.deconv3=nn.ConvTranspose2d(self.n_feature,self.n_feature,kernel_size=4 ,stride=4 )
        self.deconv4=nn.ConvTranspose2d(self.n_feature,self.n_feature,kernel_size=8 ,stride=8 )
        self.deconv5=nn.ConvTranspose2d(self.n_feature,self.n_feature,kernel_size=16,stride=16)
        
        #self.dilation=[1,3,5,8,16]
        #self.atrous1=nn.Sequential(nn.Conv2d(4*16,4*16,kernel_size=3,dilation=self.dilation[0],padding=self.dilation[0]),)
        #self.atrous2=nn.Sequential(nn.Conv2d(4*16,4*16,kernel_size=3,dilation=self.dilation[1],padding=self.dilation[1]),)
        #self.atrous3=nn.Sequential(nn.Conv2d(4*16,4*16,kernel_size=3,dilation=self.dilation[2],padding=self.dilation[2]),)
        #self.atrous4=nn.Sequential(nn.Conv2d(4*16,4*16,kernel_size=3,dilation=self.dilation[3],padding=self.dilation[3]),)
        #self.atrous5=nn.Sequential(nn.Conv2d(4*16,4*16,kernel_size=3,dilation=self.dilation[4],padding=self.dilation[4]),)

        self.score=nn.Sequential(
                nn.Conv2d(4*self.n_feature,self.n_classes,1),
                #nn.Dropout(0.5),
                )

    def forward(self,T1,IR,T2):
        T1_conv1=self.conv_block1(T1)
        T1_conv2=self.conv_block2(self.pool(T1_conv1))
        T1_conv3=self.conv_block3(self.pool(T1_conv2))
        T1_conv4=self.conv_block4(self.pool(T1_conv3))
        #T1_conv5=self.conv_block5(self.pool(T1_conv4))
        
        IR_conv1=self.conv_block1(IR)
        IR_conv2=self.conv_block2(self.pool(IR_conv1))
        IR_conv3=self.conv_block3(self.pool(IR_conv2))
        IR_conv4=self.conv_block4(self.pool(IR_conv3))
        #IR_conv5=self.conv_block5(self.pool(IR_conv4))

        T2_conv1=self.conv_block1(T2)
        T2_conv2=self.conv_block2(self.pool(T2_conv1))
        T2_conv3=self.conv_block3(self.pool(T2_conv2))
        T2_conv4=self.conv_block4(self.pool(T2_conv3))
        #T2_conv5=self.conv_block5(self.pool(T2_conv4))

        conv1_c=self.conv1_c(torch.cat([T1_conv1,IR_conv1,T2_conv1],1))
        conv2_c=self.conv2_c(torch.cat([T1_conv2,IR_conv2,T2_conv2],1))
        conv3_c=self.conv3_c(torch.cat([T1_conv3,IR_conv3,T2_conv3],1))
        conv4_c=self.conv4_c(torch.cat([T1_conv4,IR_conv4,T2_conv4],1))
        #conv5_c=self.conv5_c(torch.cat([T1_conv5,IR_conv5,T2_conv5],1))
        
        deconv2=self.deconv2(conv2_c)
        deconv3=self.deconv3(conv3_c)
        deconv4=self.deconv4(conv4_c)
        #deconv5=self.deconv5(conv5_c)

        cat=torch.cat([conv1_c,deconv2,deconv3,deconv4],1)
        #atrous1=self.atrous1(cat)
        #atrous2=self.atrous2(cat)
        #atrous3=self.atrous3(cat)
        #atrous4=self.atrous4(cat)
        #atrous5=self.atrous5(cat)
        #aspp=torch.cat([atrous1,atrous2,atrous3,atrous4,atrous5],1)
        
        score=self.score(cat)
        return score

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


if __name__=='__main__':
    x,y,z=torch.Tensor(4,3,256,256),torch.Tensor(4,3,256,256),torch.Tensor(4,3,256,256)
    x,y,z=Variable(x),Variable(y),Variable(z)
    print(x.shape)
    model=fcn_mul(n_classes=11)
    vgg16=models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    r=model(x,y,z)
    print(r.shape)


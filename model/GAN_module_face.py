import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
import torch.utils.data as Data
#import celebadataset as celebadataset
import torchvision.transforms as transforms

def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d')!=-1:
        init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('ConvTranspose2d')!=-1:
        init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('Linear')!=-1:
        init.normal_(m.weight.data,0.0,0.02)
        init.constant_(m.bias.data,0.0)
#    elif classname.find('Norm2d')!=-1:
        #init.normal_(m.weight.data,1.0,0.02)


def init_weights(nets):
    for net in nets:
        net.apply(weight_init_normal)


class Discriminator(nn.Module):
    def __init__(self,img_size=128,conv_dim=64,num_emotion=4,num_id=10177,repeat_num=6,norm='in',activ='lrelu'):
        super(Discriminator,self).__init__()

        self.model=[]
        #128*128*3->64*64*64
        self.model+=[Conv2dBlock(3, conv_dim, kernel_size=4, stride=2, padding=1,norm='in',activation=activ)]
        #downsampling blocks
        for i in range(repeat_num-1):
            #num_layer=6,64*64*64->32*32*128->16*16*256->8*8*512->4*4*1024->2*2*2048
            self.model+=[Conv2dBlock(conv_dim,2*conv_dim,kernel_size=4, stride=2, padding=1,norm=norm,activation=activ)]
            conv_dim*=2
        self.model=nn.Sequential(*self.model)

        k_s=int(img_size/np.power(2,repeat_num))

        #2*2*1024->2*2*1
        self.conv1=nn.Conv2d(conv_dim,1,kernel_size=2,stride=1,padding=0,bias=True)

    def forward(self,x):
        x=self.model(x)
        out_img_judge=self.conv1(x)
        return out_img_judge

class Classifier(nn.Module):
    def __init__(self,img_size=128,conv_dim=64,num_emotion=4,num_id=10177,repeat_num=6,norm='in',activ='lrelu'):
        super(Classifier,self).__init__()

        self.model=[]
        #128*128*3->64*64*64
        self.model+=[Conv2dBlock(3, conv_dim, kernel_size=4, stride=2, padding=1,norm='in',activation=activ)]
        #downsampling blocks
        for i in range(repeat_num-1):
            #num_layer=5,64*64*64->32*32*128->16*16*256->8*8*512->4*4*1024->2*2*2048
            self.model+=[Conv2dBlock(conv_dim,2*conv_dim,kernel_size=4, stride=2, padding=1,norm=norm,activation=activ)]
            conv_dim*=2
        self.model=nn.Sequential(*self.model)

        k_s=int(img_size/np.power(2,repeat_num))

        #2*2*2048->1*1*4
        self.conv3=nn.Conv2d(conv_dim,num_emotion,kernel_size=k_s,bias=True)

    def forward(self,x):
        x=self.model(x)
        out_img_emotion=self.conv3(x)
        return out_img_emotion.view(out_img_emotion.size(0),out_img_emotion.size(1))



class NTimesTanh(nn.Module):
    def __init__(self,n):
        super(NTimesTanh,self).__init__()
        self.n=n
        self.tanh=nn.Tanh()

    def forward(self,x):
        return self.tanh(x)*self.n




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.main=nn.ModuleList([
            #128*128*3->64*64*64->32*32*128->16*16*256->8*8*512->4*4*1024
            nn.Sequential(Conv2dBlock(3, 64, kernel_size=3, stride=2, padding=1,norm='in',activation='lrelu')),
            nn.Sequential(Conv2dBlock(64, 128, kernel_size=3, stride=2, padding=1,norm='in',activation='lrelu')),
            nn.Sequential(Conv2dBlock(128, 256, kernel_size=3, stride=2, padding=1,norm='in',activation='lrelu')),
            nn.Sequential(Conv2dBlock(256, 512, kernel_size=3, stride=2, padding=1,norm='in',activation='lrelu')),
            nn.Sequential(Conv2dBlock(512, 512, kernel_size=3, stride=2, padding=1,norm='in',activation='lrelu'))
        ])
        #4*4*1024->1*1*1024
        self.avg=nn.Sequential(Conv2dBlock(512, 1024, kernel_size=4, stride=1, padding=0,norm='none',activation='lrelu'))

        # init_weights(self.main)
        # #64*64*3->16*16*256
        # self.face_enc=FaceEncoder(num_downsample=2,num_resnet=4,input_dim=3,output_dim=64,norm='in',activ='relu')
        # #64*64*3->1*1*8
        # self.exp_enc=ExpressionEncoder(num_downsample=4,input_dim=3,output_dim=64,exp_dim=128,norm='none',activ='relu')

    def forward(self,x):
        face_info=[]
        exp_info=x
        for i in range(len(self.main)):
            exp_info=self.main[i](exp_info)
            if(i<len(self.main)-1):
                face_info.append(exp_info)
        #exp_info=self.avg(exp_info)
        return face_info,exp_info


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        #1*1*1024->4*4*1024
        #self.conv=nn.Sequential(ConvTranspose2dBlock(1024, 512, kernel_size=4, stride=1, padding=0,norm='in',activation='relu'))
        self.main=nn.ModuleList([
            #4*4*1024->8*8*512->16*16*256->32*32*128->64*64*64->128*128*3
            nn.Sequential(ConvTranspose2dBlock(512, 512, kernel_size=4, stride=2, padding=1,norm='in',activation='relu')),
            nn.Sequential(ConvTranspose2dBlock(512, 256, kernel_size=4, stride=2, padding=1,norm='in',activation='relu')),
            nn.Sequential(ConvTranspose2dBlock(256, 128, kernel_size=4, stride=2, padding=1,norm='in',activation='relu')),
            nn.Sequential(ConvTranspose2dBlock(128, 64, kernel_size=4, stride=2, padding=1,norm='in',activation='relu')),
            nn.Sequential(ConvTranspose2dBlock(64, 3, kernel_size=4, stride=2, padding=1,norm='none',activation='none')),
        ])
        # self.conv_out=nn.Conv2d(64,3,kernel_size=7,stride=1,padding=3,bias=True)
        self.activ=NTimesTanh(1)
        # init_weights(self.main)


    def forward(self,face_info,exp_info):
        x=exp_info#self.conv(exp_info)
        for i in range(len(self.main)):
            x=self.main[i](x)
            if face_info is not None and i < len(face_info):
                x=x+face_info[-i-1]
        # x=self.conv_out(x)
        return self.activ(x)

class Conv2dBlock(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding=0,norm='none',activation='relu'):
        super(Conv2dBlock,self).__init__()
        self.use_bias=True
        self.pad=nn.ReflectionPad2d(padding)

        norm_dim=output_dim

        if norm=='bn':
            self.norm=nn.BatchNorm2d(norm_dim)
        elif norm=='in':
            self.norm=nn.InstanceNorm2d(norm_dim)
        elif norm=='adain':
            self.norm=AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        else:
            self.norm=None

        if activation=='relu':
            self.activation=nn.ReLU(inplace=True)
        elif activation=='lrelu':
            self.activation=nn.LeakyReLU(0.2,inplace=True)
        elif activation=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None

        self.conv=nn.Conv2d(input_dim,output_dim,kernel_size,stride,bias=self.use_bias)

    def forward(self,x):
        x=self.conv(self.pad(x))
        if(self.norm):
            x=self.norm(x)
        if(self.activation):
            x=self.activation(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding=0,norm='none',activation='relu'):
        super(ConvTranspose2dBlock,self).__init__()
        self.use_bias=True
        # self.pad=nn.ReflectionPad2d(padding)

        norm_dim=output_dim

        if norm=='bn':
            self.norm=nn.BatchNorm2d(norm_dim)
        elif norm=='in':
            self.norm=nn.InstanceNorm2d(norm_dim)
        elif norm=='adain':
            self.norm=AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        else:
            self.norm=None

        if activation=='relu':
            self.activation=nn.ReLU(inplace=True)
        elif activation=='lrelu':
            self.activation=nn.LeakyReLU(0.2,inplace=True)
        elif activation=='tanh':
            self.activation=nn.Tanh()
        else:
            self.activation=None

        self.conv=nn.ConvTranspose2d(input_dim,output_dim,kernel_size,stride,padding,0)

    def forward(self,x):
        x=self.conv(x)
        if(self.norm):
            x=self.norm(x)
        if(self.activation):
            x=self.activation(x)
        return x







#
#
# class FaceEncoder(nn.Module):
#     def __init__(self,num_downsample=3,num_resnet=4,input_dim=3,output_dim=64,norm='in',activ='relu'):
#         super(FaceEncoder,self).__init__()
#         self.model=[]
#         #64*64*3->64*64*64
#         self.model+=[Conv2dBlock(input_dim, output_dim, kernel_size=7, stride=1, padding=3,norm=norm,activation=activ)]
#         #downsampling blocks
#         for i in range(num_downsample):
#             #num_downsample=2,64*64*64->32*32*128->16*16*256
#             self.model+=[Conv2dBlock(output_dim,2*output_dim,kernel_size=4, stride=2, padding=1,norm=norm,activation=activ)]
#             output_dim*=2
#         #residual blocks,16*16*256->16*16*256
#         self.model+=[ResBlocks(num_resnet, output_dim,norm=norm, activation=activ)]
#         self.model=nn.Sequential(*self.model)
#         #256
#         self.output_dim=output_dim
#
#     def forward(self,x):
#         return self.model(x)
#
#
# class ExpressionEncoder(nn.Module):
#     def __init__(self,num_downsample=5,input_dim=3,output_dim=64,exp_dim=128,norm='none',activ='relu'):
#         super(ExpressionEncoder,self).__init__()
#         self.model=[]
#         #64*64*3->64*64*64
#         self.model+=[Conv2dBlock(input_dim,output_dim,kernel_size=7,stride=1,padding=3,norm=norm,activation=activ)]
#         for i in range(2):
#             #64*64*64->32*32*128->16*16*256
#             self.model+=[Conv2dBlock(output_dim,2*output_dim,kernel_size=4, stride=2, padding=1,norm=norm,activation=activ)]
#             output_dim*=2
#         for i in range(num_downsample-2):
#             #num_downsample=4,16*16*256->8*8*256->4*4*256
#             self.model+=[Conv2dBlock(output_dim,output_dim,kernel_size=4, stride=2, padding=1,norm=norm,activation=activ)]
#         #global avg pooliing,4*4*256->1*1*256
#         self.model+=[nn.AdaptiveAvgPool2d(1)]
#         #1*1*256->1*1*128
#         self.model+=[nn.Conv2d(output_dim,exp_dim,kernel_size=1, stride=1, padding=0)]
#         self.model=nn.Sequential(*self.model)
#         self.output_dim=output_dim
#
#     def forward(self,x):
#         return self.model(x)
#
# class Interpolate(nn.Module):
#     def __init__(self,size,scale_factor=2,mode='nearest'):
#         self.interp=nn.functional.interpolate
#         self.scale_factor=scale_factor
#         self.size=size
#         self.mode=mode
#
#     def forward(self,x):
#         x=self.interp(x,size=self.size,scale_factor=self.scale_factor, mode=self.mode,align_corners=False)
#         return x
#
#
# class FaceDecoder(nn.Module):
#     def __init__(self,num_upsample=3,num_resnet=4,input_dim=256,output_dim=3,res_norm='adain',activ='relu'):
#         super(FaceDecoder,self).__init__()
#
#         self.model=[]
#         #AdaIN residual blocks,16*16*256
#         self.model+=[ResBlocks(num_resnet, input_dim,norm=res_norm, activation=activ)]
#         #upsampling
#         for i in range(num_upsample):
#             #num_upsample=2
#             #16*16*256->32*32*256->32*32*128
#             #32*32*128->64*64*128->64*64*64
#             self.model+=[nn.Upsample(scale_factor=2),Conv2dBlock(input_dim,input_dim//2,kernel_size=5, stride=1, padding=2,norm='ln',activation=activ)]
#             input_dim//=2
#         #now input_dim=64
#         #use reflection padding
#         #64*64*64->64*64*3
#         self.model+=[Conv2dBlock(input_dim,output_dim,kernel_size=7, stride=1, padding=3,norm='none',activation='tanh')]
#         self.model=nn.Sequential(*self.model)
#
#     def forward(self,x):
#         return self.model(x)
#
#
# class ResBlock(nn.Module):
#     def __init__(self, dim, norm='in', activation='relu'):
#         super(ResBlock, self).__init__()
#         model = []
#         model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation)]
#         model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none')]
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         residual = x
#         out = self.model(x)
#         out += residual
#         return out
#
#
# class ResBlocks(nn.Module):
#     def __init__(self, num_blocks, dim, norm='in', activation='relu'):
#         super(ResBlocks, self).__init__()
#         self.model = []
#         for i in range(num_blocks):
#             self.model += [ResBlock(dim, norm=norm, activation=activation)]
#         self.model = nn.Sequential(*self.model)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, dim=256, n_blk=3, norm='none', activ='relu'):
#         super(MLP, self).__init__()
#         self.model = []
#         #1*1*8->1*1*256
#         self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
#         for i in range(n_blk - 2):
#             #1*1*256->1*1*256
#             self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
#         #1*1*256->1*1*output_dim
#         self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
#         self.model = nn.Sequential(*self.model)
#
#     def forward(self, x):
#         return self.model(x.view(x.size(0), -1))
#
#
#
#
# class LinearBlock(nn.Module):
#     def __init__(self,input_dim,output_dim,norm='none',activation='relu'):
#         super(LinearBlock,self).__init__()
#         self.use_bias=False
#
#         norm_dim=output_dim
#
#         if norm=='bn':
#             self.norm=nn.BatchNorm2d(norm_dim)
#         elif norm=='in':
#             self.norm=nn.InstanceNorm2d(norm_dim)
#         elif norm=='adain':
#             self.norm=AdaptiveInstanceNorm2d(norm_dim)
#         elif norm == 'ln':
#             self.norm = LayerNorm(norm_dim)
#         else:
#             self.norm=None
#
#         if activation=='relu':
#             self.activation=nn.ReLU(inplace=True)
#         elif activation=='lrelu':
#             self.activation=nn.LeakyReLU(0.2,inplace=True)
#         elif activation=='tanh':
#             self.activation=nn.Tanh()
#         else:
#             self.activation=None
#
#         self.fc=nn.Linear(input_dim,output_dim,bias=self.use_bias)
#
#     def forward(self,x):
#         x=self.fc(x)
#         if(self.norm):
#             x=self.norm(x)
#         if(self.activation):
#             x=self.activation(x)
#         return x
#
#
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# if __name__ == '__main__':
#     transform_img = transforms.Compose([
#                         transforms.CenterCrop(178),
#                         transforms.Resize(64),
#                         transforms.ToTensor(),
#                         ])
#     train_dataset=celebadataset.celebadataset(img_dir='img_align_celeba',attrpath='list_attr_celeba.txt',identipath='identity_CelebA.txt',transform=transform_img,mode='train',load_data=False)
#     train_dataloader=Data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0)
#     for i,(img1,label1,id1,img2,label2,id2) in enumerate(train_dataloader):
#         encoder=Encoder()
#         decoder=Decoder()
#         discriminator=Discriminator()
#         face_info,exp_info=encoder(img1)
#         img=decoder(face_info,exp_info)
#         img_judge,img_id,img_exp=discriminator(img)
#         print(img1,label1,id1,img2,label2,id2)
#         print("train")
#         break

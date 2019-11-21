'''
PG  policy 
Progressive focus v4
把几个roialign的结果重叠起来cnn。
'''
import torchvision
import os
import re
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from maskrcnn_benchmark.layers import ROIAlign 
from fpn_resnet import resnet50 as resnet
import math

class FocusPolicy(nn.Module):
    def __init__(self):
        #3 input image channels and 12 possible actionTrue
        super(FocusPolicy,self).__init__()
        self.pool_size = 23
        self.resnet = resnet(pretrained=True,num_classes=64)  # num_classes torch.rand(10,3,256,256)
        self.conv1=nn.Conv2d(64*11,64,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(64,32,3)
        self.bn1 = nn.BatchNorm2d(64)
        #self.conv4c=nn.Conv2d(16,16,3)
        #self.fc1=nn.Linear(16*9*13+16*self.pool_size*self.pool_size,120)
        self.fc1=nn.Linear(32*(8)*(8),128)
        #self.fc2=nn.Linear(128,128)
        self.roialign = ROIAlign((self.pool_size, self.pool_size), 0.25, 2)
        self.action_head=nn.Linear(128,12)
        angle = -5*math.pi/180
        self.theta1 = torch.tensor([ [math.cos(angle),math.sin(-angle),0], 
                       [math.sin(angle),math.cos(angle) ,0] ], dtype=torch.float)
        angle = 5*math.pi/180
        self.theta2 = torch.tensor([ [math.cos(angle),math.sin(-angle),0], 
                       [math.sin(angle),math.cos(angle) ,0] ], dtype=torch.float)

        
    def forward(self,x,box):
        x = x.permute(0,3,1,2)
        x = x.float()
        x = self.resnet(x)
        theta1 = torch.cat([self.theta1.unsqueeze(0)]*x.size()[0]) 
        theta2 = torch.cat([self.theta2.unsqueeze(0)]*x.size()[0]) 
        ind = torch.from_numpy(np.asarray(range(x.size()[0]),dtype=np.float32)).view(-1,1)
        if box.is_cuda:
            ind = ind.cuda()
            theta1 = theta1.cuda()
            theta2 = theta2.cuda()
        grid1 = F.affine_grid(theta1, x.size())
        xt1 = F.grid_sample(x, grid1)   # the feature map that rotate -5 degree.
        grid2 = F.affine_grid(theta2, x.size())
        xt2 = F.grid_sample(x, grid2)   # the feature map that rotate 5 degree.
        #print("***",x.size(),xt1.size(),xt2.size(),box.size(),ind.size())
        box1=box[:,0:4]
        box2=box[:,4:8]
        box3=box[:,8:12]
        box4=box[:,12:16]
        box5=box[:,16:20]
        box6=box[:,20:24]
        box7=box[:,24:28]
        box1 = torch.cat((ind,box1), 1).detach()
        box2 = torch.cat((ind,box2), 1).detach()
        box3 = torch.cat((ind,box3), 1).detach()
        box4 = torch.cat((ind,box4), 1).detach()
        box5 = torch.cat((ind,box5), 1).detach()
        box6 = torch.cat((ind,box6), 1).detach()
        box7 = torch.cat((ind,box7), 1).detach()
        focus1 = self.roialign(x, box1)
        focus2 = self.roialign(x, box2)
        focus3 = self.roialign(x, box3)
        focus4 = self.roialign(x, box4)
        focus5 = self.roialign(x, box5)
        focus6 = self.roialign(x, box6)
        focus7 = self.roialign(x, box7)
        focus8 = self.roialign(xt1, box1) # rotation -5
        focus9 = self.roialign(xt1, box2)
        focus10 = self.roialign(xt2, box1) # rotation 5
        focus11 = self.roialign(xt2, box2)
        #x=self.pool(F.relu(self.conv1(x)))
        #x=self.pool(F.relu(self.conv2(x)))
        #print("temp:",x.size(),focus1.size())
        #x=x.view(-1,32*10*14)
        #focus1=focus1.view(-1,32*self.pool_size*self.pool_size)
        #focus2=focus2.view(-1,32*self.pool_size*self.pool_size)
        #focus3=focus3.view(-1,32*self.pool_size*self.pool_size)
        #focus4=focus4.view(-1,32*self.pool_size*self.pool_size)
        x = torch.cat((focus1,focus2,focus3,focus4,focus5,focus6,focus7,focus8,focus9,focus10,focus11), 1)
        #print("***:",x.size())
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.conv2(x))
        #print("***:",x.size())
        x=x.view(-1,32*8*8)
        #print("***:",x.size())      
        x=F.relu(self.fc1(x))
        x = torch.tanh(self.action_head(x))
        return x

if __name__ == '__main__':
    # on cpu
    img = torch.rand(20, 196, 256, 3)
    box = torch.rand(20, 28)
    pl = FocusPolicy()
    out = pl(img,box)
    print(img.size(),box.size(),out.size(),out.dtype)
    
    #exit(0)
    # on GPU
    img = torch.rand(20, 196, 256, 3).cuda(0)
    box = torch.rand(20, 28).cuda(0)
    pl = FocusPolicy().cuda(0)
    out = pl(img,box).cpu()
    print(img.size(),box.size(),out.size(),out.dtype)

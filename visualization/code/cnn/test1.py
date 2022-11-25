from torch.nn.modules.activation import Softmax
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable, backward
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import scipy.io as scio
import os


# Baseline method
class CNet(nn.Module):

    def __init__(self):
        super(CNet,self).__init__()
        self.Softmax = nn.Softmax(dim=1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,padding=1,stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 64, kernel_size=3,padding=1,stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 16, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 4, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

        )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 64, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 16, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 4, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 64, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 16, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 4, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 64, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 16, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 4, kernel_size=3,padding=1,stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 1, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 64, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 16, kernel_size=3,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 4, kernel_size=3,padding=1,stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 1, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

        )
        self.rnn = nn.LSTM(
            input_size=5632,
            hidden_size=48,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,
        )
        self.rnn1 = nn.LSTM(
            input_size=5632,
            hidden_size=48,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=4096,
            hidden_size=48,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(5*4*3*4, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 8))
    def forward(self,x,y,z,m,n):
        output1 = self.cnn(x)
        y = y.reshape(y.size()[0],y.size()[1],-1)
        m = m.reshape(m.size()[0],m.size()[1],-1)
        n = n.reshape(n.size()[0],n.size()[1],-1)
        y = y.transpose(0,1)
        m = m.transpose(0,1)
        n = n.transpose(0,1)
        output2,(h_n, h_c) = self.rnn2(y,None)
        output4,(h_n, h_c) = self.rnn(m,None)
        output5,(h_n, h_c) = self.rnn1(n,None)
        output3 = self.cnn2(z)
        output1 = output1.view(output1.size()[0], -1)
        output3 = output3.view(output3.size()[0], -1)
        output = torch.cat((output1,output2[:,-1,:],output3,output4[:,-1,:],output5[:,-1,:]),1)
        output = self.fc(output)
        output = self.Softmax(output)
        return output


# from torchviz import make_dot

def minmaxscaler(data):
    mean = data.min()
    var = data.max() 
    return (data - mean)/(var-mean)

def get_filelist(dir):
    x = np.load(dir[0])
    x = np.float32(x)
    DT = torch.from_numpy(x)
    y = np.load(dir[2])
    y = np.float32(y)
    RDT = torch.from_numpy(y)
    z = np.load(dir[1])
    z = np.float32(z)
    RT = torch.from_numpy(z)
    m = np.load(dir[3])
    m = np.float32(m)
    m = np.delete(m,[88,89,90],axis=1)
    ART = torch.from_numpy(m)
    n = np.load(dir[4])
    n = np.float32(n)
    n = np.delete(n,[88,89,90],axis=1)
    ERT = torch.from_numpy(n)
    return DT,RDT,RT,ART,ERT

def load_module():
    net = torch.load('visualization/code/cnn/net(11).pkl',map_location='cpu')
    return net

def recognize1(net, path):
    img1,img2,img3,img4,img5 = get_filelist(path)
    img1 = minmaxscaler(img1)
    img2 = minmaxscaler(img2)
    img3 = minmaxscaler(img3)
    img4 = minmaxscaler(img4)
    img5 = minmaxscaler(img5)
    img1 = torch.FloatTensor(img1.reshape([-1,1,12,64]))
    img2 = torch.FloatTensor(img2.reshape([-1,12,1,64,64]))
    img2 = img2.transpose(0,1)
    img3 = torch.FloatTensor(img3.reshape([-1,1,48,64]))
    img4 = torch.FloatTensor(img4.reshape([-1,12,1,88,64]))
    img4 = img4.transpose(0,1)
    img5 = torch.FloatTensor(img5.reshape([-1,12,1,88,64]))
    img5 = img5.transpose(0,1)
    output = net(img1,img2,img3,img4,img5)
    # acc = acc+test(output,label)
    num = torch.argmax(output, dim=1)
    if torch.max(output)>0.5:  
        num = torch.argmax(output, dim=1)
    else:
        num = torch.ones(1)*7
 
    # num = torch.argmax(output, dim=1)
    return int(num.item())


if __name__ == '__main__':
    pass
    

    
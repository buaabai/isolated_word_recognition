# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:19:50 2018

@author: spinbjy
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.optim as optim
import os
import numpy as np

class MfccDataset(Data.Dataset):
    def __init__(self,datasetpath):
        self.audiodata = []
        self.label = []
        for name in os.listdir(datasetpath):
            for file in os.listdir(os.path.join(datasetpath,name)):
                data = np.loadtxt(os.path.join(datasetpath,name,file))
                self.audiodata.append(data)
                self.label.append(np.array(int(name)))
    def __getitem__(self,index):
        data = torch.from_numpy(self.audiodata[index].T)
        data = data.type(torch.FloatTensor)
        label = torch.from_numpy(self.label[index])
        label =label.type(torch.LongTensor)
        return data,label
    def __len__(self):
        return len(self.audiodata)



class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN,self).__init__()
        self.rnn = nn.LSTM(
                input_size = 20,
                hidden_size = 100,
                num_layers = 5,
                batch_first = True,
                )
        self.out = nn.Linear(100,10)
    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.bn1 = nn.BatchNorm2d(1)    
        self.conv1 = nn.Conv2d(1,3,kernel_size = 3)  #3x18x65
        self.bn_conv1 = nn.BatchNorm2d(3)   
        self.conv2 = nn.Conv2d(3,6,kernel_size = 3) #6x16x63
        self.bn_conv2 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d((2,3),stride=(2,3))    #6x8x21
        self.conv3 = nn.Conv2d(6,8,kernel_size = 3)  #8x6x19
        self.bn_conv3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8,10,kernel_size = 3) #10x4x17
        self.bn_conv4 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2,stride = 1) #10x3x16
        self.fc1 = nn.Linear(480,200)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(200,10)
    def forward(self,x):
        x = x.reshape(-1,1,20,67)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = self.conv4(x)
        x = self.bn_conv4(x)
        x = self.pool2(x)
        x = x.view(-1,480)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

def train(epoch_index,train_loader,model,optimizer,criterion):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = Variable(data),Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model,test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data,target = Variable(data),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).item()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100.*correct/len(test_loader.dataset)
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))

def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr  

test_dir = '.\\smallmfccdataset\\test'
train_dir = '.\\smallmfccdataset\\train'

test_dataset = MfccDataset(test_dir)
train_dataset = MfccDataset(train_dir)

TEST_BATCH_SIZE = 50
TRAIN_BATCH_SIZE = 200
learning_rate = 1e-4
lr_epoch = 50
test_loader = Data.DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE,shuffle=True)
train_loader = Data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)

#model = SimpleRNN()
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(1000):
    adjust_learning_rate(learning_rate,optimizer,epoch,lr_epoch)
    train(epoch,train_loader,model,optimizer,criterion)
    test(model,test_loader,criterion)

import torch
import torch.nn as nn
import numpy as np

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

class LeNet5ELU(nn.Module):
    def __init__(self):
        super(LeNet5ELU, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ELU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ELU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


class CertNetELU(nn.Module):
    def __init__(self):
        super(CertNetELU, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6)
        self.relu2 = nn.ELU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.relu3 = nn.ELU(inplace=True)
        self.linear1  = nn.Linear(16384, 1024)
        self.linear2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class LeNet5Ensemble(nn.Module):
    def __init__(self):
        super(LeNet5Ensemble, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ELU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ELU(inplace=True)
        self.linear2 = nn.Linear(200, 10)
        
        self.T = float(0.01) #Temperature level
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = out + float(np.sqrt(self.T))*torch.Tensor(out).normal_()
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out + float(np.sqrt(self.T))*torch.Tensor(out).normal_()
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        out = out + float(np.sqrt(self.T))*torch.Tensor(out).normal_()
        return out

class SubstituteModel(nn.Module):

    def __init__(self):
        super(SubstituteModel, self).__init__()
        self.linear1 = nn.Linear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear()
        self.relu1 = nn.ReLU(inplace=True)


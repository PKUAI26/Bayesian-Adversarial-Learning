import torch
from torch.autograd import Variable
from torch import optim, nn
import matplotlib.pyplot as plt
import shutil

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from tqdm import tqdm
from PIL import Image
import pickle
import numpy as np
import time
import os
import argparse
import numpy as np
import sys
import copy

from util import NUM_CLASSES, IMG_SIZE
from util import Utils

utils = Utils()
from skimage.transform import rotate, warp, ProjectiveTransform
import random




# In[8]:


class ConvNet(nn.Module):

    def __init__(self, in_c, out_c,
                 kernel_size,
                 padding_size='same',
                 pool_stride=2,
                 batch_norm=True):
        super().__init__()

        if padding_size == 'same':
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding_size)
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm
        self.batch_norm_2d = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.max_pool2d(nn.functional.leaky_relu(self.conv(x)))

        if self.batch_norm:
            return self.batch_norm_2d(x)
        else:
            return x


# In[9]:


class Classifier(nn.Module):
    def __init__(self, input_nbr, out_nbr):
        super(Classifier, self).__init__()
        self.input_nbr = input_nbr
        self.lin = nn.Linear(input_nbr, out_nbr)

    def forward(self, x):
        return self.lin(x)


# In[10]:


class SoftMaxClassifier(Classifier):
    def __init__(self, in_len, out_len):
        super().__init__(in_len, out_len)

    def forward(self, x):
        x = super().forward(x)
        return nn.functional.log_softmax(x)


# In[11]:


class FullyConnected(nn.Module):
    def __init__(self, input_nbr, out_nbr):
        super(FullyConnected, self).__init__()
        self.input_nbr = input_nbr
        self.lin = nn.Linear(input_nbr, out_nbr)
        self.rel = nn.LeakyReLU()
        self.dropout = nn.Dropout()

    def forward(self, input):
        return self.dropout(self.rel(self.lin(input)))


# In[12]:


class LocalizationNetwork(nn.Module):
    nbr_params = 6
    init_bias = torch.Tensor([1, 0, 0, 0, 1, 0])

    def __init__(self, conv_params, kernel_sizes,
                 input_size, input_channels=1):
        super(LocalizationNetwork, self).__init__()

        if not kernel_sizes:
            kernel_sizes = [5, 5]

        if len(kernel_sizes) != 2:
            raise Exception("Number of kernel sizes != 2")

        self.conv1 = ConvNet(input_channels, conv_params[0],
                             kernel_size=kernel_sizes[0],
                             batch_norm=False)
        self.conv2 = ConvNet(conv_params[0], conv_params[1],
                             kernel_size=kernel_sizes[1],
                             batch_norm=False)
        conv_output_size, _ = utils.get_convnet_output_size([self.conv1, self.conv2],
                                                            input_size)

        self.fc = FullyConnected(conv_output_size, conv_params[2])
        self.classifier = Classifier(conv_params[2], self.nbr_params)

        self.classifier.lin.weight.data.fill_(0)
        self.classifier.lin.bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        x = self.dropout(self.conv1(x))
        conv_output = self.dropout(self.conv2(x))
        conv_output = conv_output.view(conv_output.size()[0], -1)
        return self.classifier(self.fc(conv_output))


# In[13]:


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, params, kernel_sizes, input_size=IMG_SIZE,
                 input_channels=1):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization_network = LocalizationNetwork(params,
                                                        kernel_sizes,
                                                        input_size,
                                                        input_channels)

    def forward(self, input):
        out = self.localization_network(input)
        out = out.view(out.size()[0], 2, 3)
        grid = nn.functional.affine_grid(out, input.size())
        return nn.functional.grid_sample(input, grid)


# In[14]:


class GeneralNetwork(nn.Module):
    def __init__(self, opt):
        super(GeneralNetwork, self).__init__()

        if not opt.cnn:
            opt.cnn = '100, 150, 250, 350'
        self.kernel_sizes = [5, 3, 1]
        conv_params = list(map(int, opt.cnn.split(",")))

        self.conv1 = ConvNet(1, conv_params[0], kernel_size=self.kernel_sizes[0],
                             padding_size=0)
        self.conv2 = ConvNet(conv_params[0], conv_params[1],
                             kernel_size=self.kernel_sizes[1],
                             padding_size=0)

        conv_output_size, _ = utils.get_convnet_output_size([self.conv1, self.conv2])

        self.fc = FullyConnected(conv_output_size, conv_params[2])
        self.classifier = SoftMaxClassifier(conv_params[2], NUM_CLASSES)

        self.locnet_1 = None
        if opt.st and opt.locnet:
            params = list(map(int, opt.locnet.split(",")))
            self.locnet_1 = SpatialTransformerNetwork(params,
                                                      kernel_sizes=[7, 5])

        self.locnet_2 = None
        if opt.st and opt.locnet2:
            params = list(map(int, opt.locnet2.split(",")))
            _, current_size = utils.get_convnet_output_size([self.conv1])
            self.locnet_2 = SpatialTransformerNetwork(params,
                                                      [5, 3],
                                                      current_size,
                                                      conv_params[0])
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        if self.locnet_1:
            x = self.locnet_1(x)

        x = self.conv1(x)

        if self.locnet_2:
            x = self.locnet_2(x)

        return self.classifier(self.fc(self.dropout(self.conv2(x))))


# In[15]:


class IDSIANetwork(GeneralNetwork):
    def __init__(self, opt):
        super().__init__(opt)
        conv_params = list(map(int, opt.cnn.split(",")))

        self.conv3 = ConvNet(conv_params[1], conv_params[2], kernel_size=self.kernel_sizes[2],
                             padding_size=0)
        conv_output_size, _ = utils.get_convnet_output_size([self.conv1,
                                                      self.conv2,
                                                      self.conv3])
        self.fc = FullyConnected(conv_output_size, conv_params[3])
        self.classifier = SoftMaxClassifier(conv_params[3], NUM_CLASSES)

        self.locnet_3 = None
        if opt.st and opt.locnet3:
            params = list(map(int, opt.locnet3.split(",")))
            _, current_size = utils.get_convnet_output_size([self.conv1, self.conv2])
            self.locnet_3 = SpatialTransformerNetwork(params,
                                                      [3, 3],
                                                      current_size,
                                                      conv_params[1])

    def forward(self, x):
        if self.locnet_1:
            x = self.locnet_1(x)

        x = self.conv1(x)
        x = self.dropout(x)

        if self.locnet_2:
            x = self.locnet_2(x)

        x = self.conv2(x)
        x = self.dropout(x)

        if self.locnet_3:
            x = self.locnet_3(x)

        x = self.conv3(x)
        x = self.dropout(x)

        x = x.view(x.size()[0], -1)
        return self.classifier(self.fc(x))

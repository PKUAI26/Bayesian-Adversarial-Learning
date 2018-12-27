from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import exposure
from PIL import Image

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

import warnings 

NUM_CLASSES = 43
IMG_SIZE = 32

class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels, fixedindex=0):
        self.images = torch.from_numpy(images)
        self.images = self.images.permute(0, 3, 1, 2)
        self.labels = torch.LongTensor(labels)
        self.fixedindex = fixedindex
        if self.fixedindex != 0:
            print('warning: using only 1 indexed dataset')

    def __len__(self):
        if self.fixedindex != 0:
            return 1
        return len(self.images)

    def __getitem__(self, index):
        if self.fixedindex != 0:
            return self.images[self.fixedindex], self.labels[self.fixedindex]
        return self.images[index], self.labels[index]

class Utils:
    def __init__(self):
        self.train_data_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        self.val_data_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        
    def load_pickled_data(self, file, columns):
        with open(file, mode='rb') as f:
            dataset = pickle.load(f)
        return tuple(map(lambda c: dataset[c], columns))
    
    def get_dataset(self, params, fixedindex=0):
        if params.use_pickle:
            data_images, data_labels = self.load_pickled_data(params.train_pickle, ['features', 'labels'])
            train_images, val_images, train_labels, val_labels = train_test_split(data_images, 
                                                                                  data_labels, 
                                                                                  test_size=0.25) 
            '''
            print(type(train_images))
            print(train_images.shape)
            train_images = np.mean(train_images, axis=3)
            print(train_images.shape)
            exit(0)
            '''
            
            #print('train_images', train_images.shape)
            train_images = (np.mean(train_images, axis=3)/255.0).astype(np.float32)
            #print('train_images', train_images.shape)
            train_images = train_images[:,:,:,np.newaxis]
            val_images = (np.mean(val_images, axis=3)/255.0).astype(np.float32)
            val_images = val_images[:,:,:,np.newaxis]
            #print('train_images', train_images.shape)
            #exit(0)
            return TrafficSignsDataset(train_images, train_labels, fixedindex=fixedindex), TrafficSignsDataset(val_images, val_labels, fixedindex=fixedindex)
        else:
            train_dataset = datasets.ImageFolder(params.data + '/train_images',
                                                 transform=self.train_data_transforms)
            val_dataset = datasets.ImageFolder(self.params.data + '/val_images',
                                               transform=self.val_data_transforms)
            return train_dataset, val_dataset
    
    def pickle_data(self, x, y, save_loc):
        print("Saving pickle at " + save_loc)
        save = {"features": x, "labels": y}
        
        with open(save_loc, "wb") as f:
            pickle.dump(save, f)
    
    def pickle_data_from_folder(self, data_folder, save_loc):
        if not os.path.isdir(data_folder):
            print("Data folder must be a folder and should contains sub folders for each label")
            return
        
        resize_transform = transforms.Resize((IMG_SIZE, IMG_SIZE))
        sub_folders = os.listdir(data_folder)
        
        count = 0
        for sub_folder in sub_folders:
            sub_folder = os.path.join(data_folder, sub_folder)

            if not os.path.isdir(sub_folder):
                continue
            label = int(sub_folder.split("/")[-1])

            for image in os.listdir(sub_folder):
                count += 1

        save = {"features": np.empty([count, IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8), 
                "labels": np.empty([count], dtype=int)}
        i = 0
        for sub_folder in sub_folders:
            sub_folder = os.path.join(data_folder, sub_folder)

            if not os.path.isdir(sub_folder):
                continue
            label = int(sub_folder.split("/")[-1])
            for image in os.listdir(sub_folder):
                image = os.path.join(sub_folder, image)
                pic = Image.open(image)
                pic = resize_transform(pic)
                pic = np.array(pic)
                save["features"][i] = pic
                save["labels"][i] = label
                i += 1

        
        with open(save_loc, "wb") as f:
            pickle.dump(save, f)
    
    def get_dataset_from_file(self, file):
        data_images, data_labels = self.load_pickled_data(file, ['features', 'labels'])
        
        return TrafficSignsDataset(data_images, data_labels)

    def get_convnet_output_size(self, network, input_size=IMG_SIZE):
        input_size = input_size or IMG_SIZE

        if type(network) != list:
            network = [network]

        in_channels = network[0].conv.in_channels

        output = Variable(torch.ones(1, in_channels, input_size, input_size))
        output.require_grad = False
        for conv in network:
            output = conv.forward(output)

        return np.asscalar(np.prod(output.data.shape)), output.data.size()[2]
    def get_time_hhmmss(self, start = None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None:
            return time.strftime("%Y/%m/%d %H:%M:%S")
        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str 
    
        
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        epoch = state['epoch']
        print("=> Saving model to %s" % filename)

        if is_best:
            print("=> The model just saved has performed best on validation set" +
                  " till now")
            shutil.copyfile(filename, 'model_best.pth.tar')
        
        return filename


    def load_checkpoint(self, resume):
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            
            if not torch.cuda.is_available():
                checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
            else:
                checkpoint = torch.load(resume)
            print("=> loaded checkpoint '{}' (epoch {})"
                     .format(resume, checkpoint['epoch']))
            return checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return None
    
    
    def preprocess_dataset(self, X, y=None, use_tqdm=True):
        # Convert to single channel Y
        X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
        
        # Scale
        X = (X / 255.).astype(np.float32)
        
        # Don't want to use tqdm while generating csv
        if use_tqdm:
            preprocess_range = tqdm(range(X.shape[0]))
        else:
            preprocess_range = range(X.shape[0])
            
        # Ignore warnings, see http://scikit-image.org/docs/dev/user_guide/data_types.html
        for i in preprocess_range:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X[i] = exposure.equalize_adapthist(X[i])

        if y is not None:  
            # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
            y = np.eye(NUM_CLASSES)[y]
            X, y = shuffle(X, y)

        # Add a single grayscale channel
        X = X.reshape(X.shape + (1,)) 
        return X, y

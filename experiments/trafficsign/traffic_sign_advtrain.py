
# coding: utf-8

# In[1]:


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
sys.path.append('../../pytorch_adversarial')

from adversarialbox.attacks import FGSMAttack, PGDAttack
from adversarialbox.bayesattacks import BayesWRM
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from sampler.sghmc import SGHMC
from sampler.sgadahmc import SGAdaHMC

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

#xp = []

parser = argparse.ArgumentParser()
parser.add_argument("--advtraining", type=str, default='BayesWRM', help="adv training type")
parser.add_argument("--epsilon", type=float, default=0.05, help="adv training epsilon")
parser.add_argument("--a", type=float, default=350, help="adv learning rate for BayesWRM")
parser.add_argument("--T", type=float, default=0, help="adv Temperature for inner maximization")
parser.add_argument("--initlr", type=float, default=0.01, help="Temperature for out maximization")
parser.add_argument("--T_out", type=float, default=1e-5, help="Temperature for out maximization")
parser.add_argument("--Epoch", type=int, default=1, help="Number of Epochs for BayesWRM and Bayes")
parser.add_argument("--L", type=int, default=15, help="Number of iterations for solving maximization")
parser.add_argument("--multi", type=bool, default=False, help="Whether use multi method")
parser.add_argument("--inneroptimizer", type=str, default='SGHMC', help="optimizer to sample")
parser.add_argument("--outoptimizer", type=str, default='SGAdaHMC', help="optimizer to sample")
#WRM parameters
parser.add_argument("--eps", type=float, default=0.3, help="adv training epsilon")
#Store intermediate results
parser.add_argument("--storeadv", type=bool, default=False, help="whether store the results")
args = parser.parse_args()



if args.storeadv == True:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


NUM_CLASSES = 43
IMG_SIZE = 32


# In[3]:

advtraining = args.advtraining
print('======================================')
param = {
        'model'     :   'resnet18',
        'patience'  :   5,
        'batch_size':   64,
        'nepochs'   :   10,
        'nworkers'  :   4,
        'seed'      :   1,
        'data'      :   'fashion',
        'epsilon'   :   args.epsilon,
        }


if advtraining == 'FGSM':
    adversary = FGSMAttack(epsilon=param['epsilon'], storeadv=args.storeadv)
    T = 0.0
    print('use FGSM adv training')
elif advtraining == 'IFGSM':
    adversary = PGDAttack(epsilon=param['epsilon'], k=15,order='inf', storeadv=args.storeadv)
    T = 0.0
    print('use LinfPGD adv training')

elif advtraining == 'PGD':
    #adversary = LinfPGDAttack(epsilon=param['epsilon'], k=1, order='2')
    adversary = PGDAttack(epsilon=param['epsilon'], k=1, order='2', storeadv=args.storeadv)
    T = 0.0
    print('use PGD advtraining')


elif advtraining == 'ERM':
    adversary = None
    T = 0.0
    print('use ERM training')

elif advtraining == 'BayesWRM':
    Sz = 5
    Stheta = 5
    gamma = 0.05 #2
    a = args.a
    T = args.T
    #adversary = BayesWRM(a=1e-2, k=15, epsilon=10, gamma=gamma, T=1e-1, Sz=Sz, Stheta=Stheta)
    adversary = BayesWRM(a=a, k=args.L, epsilon=args.epsilon, gamma=gamma, T=T, Sz=Sz, Stheta=Stheta, optimizer = args.inneroptimizer, clip=False, proj=None, storeadv=args.storeadv)
elif advtraining == 'Bayes':
    adversary = None
    T = 0.0
    Stheta = 5
    print('use Bayes training')
else:
    raise NotImplementedError("Adv attack type note implemented yet")


print('advtraining method', advtraining)
params = Namespace()
params.data = './data'
params.lr = 0.0001
params.batch_size = 256 #128
params.seed = 7
params.cnn = '50, 100, 150, 350'
params.locnet = '10,10,10'
params.locnet2 = None
params.locnet3 = None
params.st = True
params.resume = False
if advtraining == 'BayesWRM' or advtraining == 'Bayes':
    params.epochs = args.Epoch #15
    params.patience = args.Epoch #

elif advtraining == 'WRM':
    params.epochs = 5
    params.patience = 5

else:
    params.epochs = 1
    params.patience = 1

params.dropout = 0.5
params.use_pickle = True
params.save_loc = "."
params.outfile = 'gtsrb_kaggle.csv'
#params.train_pickle = params.save_loc + '/train_balanced_preprocessed.p'
params.train_pickle = params.save_loc + '/train.p'
params.extra_debug = False


from util import Utils
utils = Utils()



from model import IDSIANetwork
# In[16]:





class Trainer:
    def __init__(self, params, train_data=None, val_data=None):
        self.params = params
        self.train_data = train_data
        self.val_data = val_data

        print("Creating dataloaders")
        self.cuda_available = torch.cuda.is_available()

        if self.train_data is not None:
            self.train_loader = DataLoader(dataset=self.train_data,
                                           shuffle=True,
                                           batch_size=params.batch_size,
                                           pin_memory=self.cuda_available)
        if self.val_data is not None:
            self.val_loader = DataLoader(dataset=self.val_data,
                                         shuffle=False,
                                         batch_size=params.batch_size,
                                         pin_memory=self.cuda_available)

        self.string_fixer = "=========="

    def load(self):
        print("Loading model")
        
        if advtraining == 'BayesWRM' or advtraining == 'Bayes':
            net = IDSIANetwork(self.params)
            net.load_state_dict(torch.load('ERM.model'))
            self.model_list = [copy.deepcopy(net) for i in range(Stheta)]
            for net in self.model_list:
                net.cuda()
        else:
            self.model = IDSIANetwork(self.params)
            #if advtraining == 'FGSM':
            #For all methods init from ERM trained model
            self.model.load_state_dict(torch.load('ERM.model'))
                
            if self.cuda_available:
                self.model = self.model.cuda()
            
            '''
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.params.lr)

            '''
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                               self.model.parameters()),
                                        lr=self.params.lr)

        self.start_time = time.time()
        self.histories = {
            "train_loss": np.empty(0, dtype=np.float32),
            "train_acc": np.empty(0, dtype=np.float32),
            "val_loss": np.empty(0, dtype=np.float32),
            "val_acc": np.empty(0, dtype=np.float32)
        }

        # We minimize the cross entropy loss here
        
        '''
        self.early_stopping = EarlyStopping(
            self.model, self.optimizer, params=self.params,
            patience=self.params.patience, minimize=True)
        '''
        if self.params.resume:
            checkpoint = utils.load_checkpoint(self.params.resume)
            if checkpoint is not None:

                if "params" in checkpoint:
                    # To make sure model architecture remains same
                    self.params.locnet = checkpoint['params'].locnet
                    self.params.locnet2 = checkpoint['params'].locnet2
                    self.params.locnet3 = checkpoint['params'].locnet3
                    self.params.st = checkpoint['params'].st

                    self.model = IDSIANetwork(self.params)
                    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                                       self.model.parameters()),
                                                lr=self.params.lr)

                self.model.load_state_dict(checkpoint['state_dict'])
                
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                SELF.histories.update(checkpoint)
                #self.early_stopping.init_from_checkpoint(checkpoint)
                print("Loaded model, Best Loss: %.8f, Best Acc: %.2f" %
                      (checkpoint['best'], checkpoint['best_acc']))


    def train(self):
        self.epochs = self.params.epochs

        criterion = nn.CrossEntropyLoss()
        start_epoch = 0

        if advtraining == 'BayesWRM' or advtraining == 'Bayes':
            for net in self.model_list:
                net.train()
        else:
            self.model.train()
        
        print("Starting training")
        self.print_info()

        init_lr = 0.5
        init_lr = args.initlr

        for epoch in range(start_epoch, self.params.epochs):
            print('start epoch', str(epoch))
            print('advtraining method', advtraining)
            #break
            for i, (images, labels) in enumerate(self.train_loader):
                

                X, y = images.cuda(), labels.cuda()
                x_var, y_var = to_var(X), to_var(y)
                
                if adversary is not None:
                    x_adv = X.cpu()
                    
                    if advtraining == 'BayesWRM':
                        x_adv = adv_train(X=x_adv, y=labels.cpu().long(), model=self.model_list, criterion=criterion, adversary=adversary)   
                        
                        for i in range(Stheta):
                            x_adv_temp = x_adv
                            if args.multi == True:
                                del x_adv_temp[i]

                            if epoch < 2:
                                lr = init_lr
                            elif epoch < 5:
                                lr = 0.1*init_lr
                            elif epoch < 10:
                                lr = 0.1*init_lr
                            else:
                                lr = 0.05*init_lr
                            #optimizer = SGHMC(self.model_list[i].parameters(), config=dict(lr=lr))
                            if args.outoptimizer == 'SGHMC':
                                optimizer = SGHMC(filter(lambda x: x.requires_grad, self.model_list[i].parameters()), config=dict(lr=lr, T=args.T_out))
                            elif args.outoptimizer == 'SGAdaHMC':
                                optimizer = SGAdaHMC(filter(lambda x: x.requires_grad, self.model_list[i].parameters()), config=dict(lr=0.01, T=args.T_out))
                            else:
                                raise NotImplementedError('Inner optimizer not implemented')

                            def helper():
                                def feval():
                                    loss_adv = 0
                                    for k in range(len(x_adv_temp)):
                                        x_adv_var = to_var(torch.from_numpy(x_adv_temp[k].astype(np.float32)))
                                        #loss_adv = loss_adv + criterion(net(x_adv_var), y_var)
                                        #add adversarial loss
                                        loss_adv = loss_adv + criterion(self.model_list[i](x_adv_var), y_var)
                                        #add clean loss
                                        loss_adv = loss_adv + criterion(self.model_list[i](x_var), y_var)
                                    loss_adv = loss_adv/2.0
                                    
                                    optimizer.zero_grad()
                                    loss_adv.backward()
                                
                                    return loss_adv #TODO return loss for extension
                                return feval
                            #Tracer()()
                            loss_adv = optimizer.step(helper())
                            #print('Epoch:', epoch, 'model:', i, 'loss', loss_adv.data.cpu().numpy()[0])
                        #print("Current timestamp: %s" % (utils.get_time_hhmmss()))


                    else:    
                        x_adv = adv_train(x_adv, y.cpu().long(), self.model, criterion, adversary)
                
                        x_adv_var = to_var(x_adv)
                        loss_adv = criterion(self.model(x_adv_var), y_var)
                        loss = (loss_adv + criterion(self.model(x_var),  y_var))/2.0

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                
                else:
                    if advtraining == 'Bayes':
                        for i in range(Stheta):
                            if epoch < 2:
                                lr = init_lr
                            elif epoch < 5:
                                lr = 0.1*init_lr
                            elif epoch < 10:
                                lr = 0.1*init_lr
                            else:
                                lr = 0.05*init_lr
                            #optimizer = SGHMC(self.model_list[i].parameters(), config=dict(lr=lr))
                            if args.outoptimizer == 'SGHMC':
                                optimizer = SGHMC(filter(lambda x: x.requires_grad, self.model_list[i].parameters()), config=dict(lr=lr, T=args.T_out))
                            elif args.outoptimizer == 'SGAdaHMC':
                                optimizer = SGAdaHMC(filter(lambda x: x.requires_grad, self.model_list[i].parameters()), config=dict(lr=0.01, T=args.T_out))
                            else:
                                raise NotImplementedError('Outer optimizer not implemented')

                            def helper():
                                def feval():
                                    loss_adv =  criterion(self.model_list[i](x_var), y_var)
                                    optimizer.zero_grad()
                                    loss_adv.backward()
                                
                                    return loss_adv #TODO return loss for extension
                                return feval
                            #Tracer()()
                            loss = optimizer.step(helper())
                    else:
                        loss = criterion(self.model(x_var), y_var)
                    
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()


                if self.params.extra_debug and (i + 1) % (self.params.batch_size * 4) == 0:
                    print(('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4},')
                          .format(epoch + 1,
                                  self.params.epochs,
                                  i + 1,
                                  len(self.train_loader),
                                  loss.data[0]))

            print('entering validation loss set the advtraining method is ', advtraining)
            if  advtraining == 'BayesWRM' or advtraining == 'Bayes':
                train_acc, train_loss = self.validate_model(self.train_loader, self.model_list[0])
                val_acc, val_loss = self.validate_model(self.val_loader, self.model_list[0])
            else:
                train_acc, train_loss = self.validate_model(self.train_loader, self.model)
                val_acc, val_loss = self.validate_model(self.val_loader, self.model)
                

            self.histories['train_loss'] = np.append(self.histories['train_loss'], [train_loss])
            self.histories['val_loss'] = np.append(self.histories['val_loss'], [val_loss])
            self.histories['val_acc'] = np.append(self.histories['val_acc'], [val_acc])
            self.histories['train_acc'] = np.append(self.histories['train_acc'], [train_acc])
            print('trianacc', str(train_acc), 'valacc', str(val_acc))

            print('advtraining method', advtraining)

    def validate_model(self, loader, model):
        model = copy.deepcopy(model)
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

        for images, labels in loader:
            images_batch = Variable(images, volatile=True)
            labels_batch = Variable(labels.long())

            if self.cuda_available:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()

            output = model(images_batch)
            loss = nn.functional.cross_entropy(output, labels_batch.long(), size_average=False)
            total_loss += loss.data[0]
            total += len(labels_batch)

            if not self.cuda_available:
                correct += (labels_batch == output.max(1)[1]).data.cpu().numpy().sum()
            else:
                correct += (labels_batch == output.max(1)[1]).data.sum()
        model.train()

        average_loss = total_loss / total
        return correct / total * 100, average_loss

    def print_info(self):
        print(self.string_fixer + " Data " + self.string_fixer)
        print("Training set: %d examples" % (len(self.train_data)))
        print("Validation set: %d examples" % (len(self.val_data)))
        print("Timestamp: %s" % utils.get_time_hhmmss())

        print(self.string_fixer + " Params " + self.string_fixer)

        print("Learning Rate: %f" % self.params.lr)
        print("Dropout (p): %f" % self.params.dropout)
        print("Batch Size: %d" % self.params.batch_size)
        print("Epochs: %d" % self.params.epochs)
        print("Patience: %d" % self.params.patience)
        print("Resume: %s" % self.params.resume)

    def print_train_info(self, epoch, train_acc, train_loss, val_acc, val_loss):
        print((self.string_fixer + " Epoch: {0}/{1} " + self.string_fixer)
              .format(epoch + 1, self.params.epochs))
#         print("Train Loss: %.8f, Train Acc: %.2f" % (train_loss, train_acc))
        print("Validation Loss: %.8f, Validation Acc: %.2f" % (val_loss, val_acc))
        #self.early_stopping.print_info()
        print("Elapsed Time: %s" % (utils.get_time_hhmmss(self.start_time)))
        print("Current timestamp: %s" % (utils.get_time_hhmmss()))


# In[19]:


train_dataset, val_dataset = utils.get_dataset(params)


# Train first time with train pickle mentioned in params, should be extended to let the model now about the real distribution

# In[20]:


params.resume = False
params.extra_debug = False
trainer = Trainer(params, train_dataset, val_dataset)
trainer.load()


# In[21]:


trainer.train()



#print('mean of xp', np.mean(xp))
#exit(0)


"""
Save the trained models
"""

from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack
from adversarialbox.utils import to_var, pred_batch, test,     attack_over_test_data

#from models import LeNet5
import pandas as pd         
# Hyper-parameters
param = {
    'test_batch_size': 200,
    'epsilon': 0.001,
}

epsilon_set = np.linspace(0, 0.3, 11)
advacc_set = []

from copy import deepcopy
import os



if advtraining == 'BayesWRM' or advtraining == 'Bayes':
    model_list = trainer.model_list
    for ii, net in enumerate(model_list):
        torch.save(net.state_dict(), os.path.join('advtrained_models', args.advtraining+'_'+str(ii)+'.model'))
        for p in net.parameters():
            p.requires_grad=False
        net.eval()
else:
    net = trainer.model
    torch.save(net.state_dict(), os.path.join('advtrained_models', args.advtraining+'.model'))
    for p in net.parameters():
        p.requires_grad = False
        net.eval()


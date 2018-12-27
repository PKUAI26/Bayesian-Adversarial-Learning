# coding: utf-8
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from fashionmnist import FashionMNIST
from torchvision import datasets
import model
import utils
import time
import argparse
import os
import sys
import copy
import numpy as np
sys.path.append('../../pytorch_adversarial')

from adversarialbox.attacks import FGSMAttack, CertAttack, WRMAttack, PGDAttack, CWAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from adversarialbox.bayesattacks import BayesWRM
from sampler.sghmc import SGHMC
from sampler.sgadahmc import SGAdaHMC


"""
Adversarial attacks on Fashion MNIST
"""
from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, PGDAttack
from adversarialbox.utils import to_var, pred_batch, test,     attack_over_test_data

#from models import LeNet5
import pandas as pd         

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

train_transforms = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # utils.RandomRotation(),
                        # utils.RandomTranslation(),
                        # utils.RandomVerticalFlip(),
                        transforms.ToTensor()
                        # transforms.Normalize((0.1307,), (0.3081,))
                        ]
                        )
val_transforms = transforms.Compose([
                        transforms.ToTensor()
                        # transforms.Normalize((0.1307,), (0.3081,))
                        ])

# Hyper-parameters
param = {
    'model':'SimpleNet',
    'test_batch_size': 100,
    'epsilon': 0.001,
}



method_set = ['BayesWRM', 'FGSM', 'IFGSM', 'PGD', 'WRM2', 'Bayes', 'ERM']


for method in method_set:
    if method == 'BayesWRM' or method == 'Bayes':
        import model
        model_list = []
        for ind in range(5):
            net = model.__dict__[param['model']]()     
            net.load_state_dict(torch.load(os.path.join('advtrained_models', method+str(ind)+'.model')))
            net.cuda()
            model_list.append(net)

            
    else:
        import model
        net = model.__dict__[param['model']]()     
        net.load_state_dict(torch.load(os.path.join('advtrained_models', method+'.model')))
        net.cuda()




    advacc_set = []

    from copy import deepcopy

                
    if method == 'BayesWRM' or method=='Bayes':
        for net in model_list:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()
    else:
        for p in net.parameters():
            p.requires_grad = False
        net.eval()

    
    steps_set = [1]

    for steps in steps_set:
        # Data loaders
        valset = FashionMNIST('data', train=False, transform=val_transforms)
        loader_test = torch.utils.data.DataLoader(valset, 
            batch_size=param['test_batch_size'], shuffle=False)    
        #test(net, loader_test)

        if method == 'BayesWRM' or method == 'Bayes':
            adversary = CWAttack(model_list, steps=steps,  advtraining='FGSM')

            advacc = attack_over_test_data(model_list, adversary, param, loader_test)
        else:
            adversary = CWAttack(net, steps=steps,  advtraining=method)

            advacc = attack_over_test_data(net, adversary, param, loader_test)
        
        print('method',method,  'adv accuracy', advacc)
        advacc_set.append(advacc)

    df = pd.DataFrame(
        {'k': list(steps_set),
         'advacc': advacc_set})

    df.to_csv(os.path.join('fashionmnistwhiteboxCW', method+'_whitebox.csv'))


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

parser = argparse.ArgumentParser()
parser.add_argument("--blackbox", type=bool, default=False, help="whether use blackbox")
parser.add_argument("--sample", type=int, default=5, help="Number of samples")
args = parser.parse_args()






method_set = ['BayesWRM', 'FGSM', 'IFGSM', 'PGD', 'Bayes', 'ERM']
#method_set = ['BayesWRM']
#method_set = ['BayesWRM']

for method in method_set:
    if method == 'BayesWRM' or method == 'Bayes':
        import model
        model_list = []
        for ind in range(args.sample):
            net = model.__dict__[param['model']]()     
            net.load_state_dict(torch.load(os.path.join('advtrained_models', method+str(ind)+'.model')))
            net.cuda()
            model_list.append(net)

            
    else:
        import model
        net = model.__dict__[param['model']]()     
        net.load_state_dict(torch.load(os.path.join('advtrained_models', method+'.model')))
        net.cuda()




    epsilon_set = np.linspace(0, 0.15, 10)
    #epsilon_set = [0.01]
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


    valset = datasets.MNIST('data-mnist', train=False, transform=val_transforms)
    loader_test = DataLoader(valset, batch_size=param['test_batch_size'],
                            shuffle=False)
    test(model_list, loader_test)
    
    for epsilon in epsilon_set:
        # Data loaders

        if method == 'BayesWRM' or method == 'Bayes':
            adversary = FGSMAttack(model_list, epsilon, is_train=False, advtraining='Bayes')
            
            advacc = attack_over_test_data(model_list, adversary, param, loader_test)
        else:
            
            adversary = FGSMAttack(net, epsilon, is_train=False, advtraining=method)
            
            advacc = attack_over_test_data(net, adversary, param, loader_test)
        
        print('method',method,  'adv accuracy', advacc)
        advacc_set.append(advacc)

    df = pd.DataFrame(
        {'epsilon': list(epsilon_set),
         'advacc': advacc_set})


    
    df.to_csv(os.path.join('mnistwhitebox', method+'_whitebox.csv'))
    

# coding: utf-8
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

from adversarialbox.attacks import FGSMAttack, CertAttack, WRMAttack, PGDAttack, CWAttack
from adversarialbox.bayesattacks import BayesWRM
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from sampler.sghmc import SGHMC
from sampler.sgadahmc import SGAdaHMC

from time import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack
from adversarialbox.utils import to_var, pred_batch, test,     attack_over_test_data
import pandas as pd
from model import IDSIANetwork


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

inputdist='Gauss'
noiseratio = 5e-2 #1e-5 #0.1

#inputdist='uniform'
#noiseratio = 0.1

param = {
    'test_batch_size': 200,
    'epsilon': 0.001,
}

#loading the whitebox model
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
params.dropout = 0.5
params.use_pickle = True
params.save_loc = "."
params.outfile = 'gtsrb_kaggle.csv'
#params.train_pickle = params.save_loc + '/train_balanced_preprocessed.p'
params.train_pickle = params.save_loc + '/train.p'
params.extra_debug = False

from util import Utils
utils = Utils()

method_set = ['BayesWRM', 'IFGSM', 'FGSM', 'PGD',  'Bayes', 'ERM']


for method in method_set:

    if method == 'BayesWRM' or method == 'Bayes':
        model_list = []
        for ii in range(5):
            net = IDSIANetwork(params)
            net.load_state_dict(torch.load(os.path.join('advtrained_models', method+'_'+str(ii)+'.model')))
            net.cuda()
            model_list.append(net)
    else:
        net = IDSIANetwork(params)
        net.load_state_dict(torch.load(os.path.join('advtrained_models', method+'.model')))
        net.cuda()

    steps_set = [1]
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

    for steps in steps_set:

        train_dataset, val_dataset = utils.get_dataset(params)
        
        loader_test = torch.utils.data.DataLoader(val_dataset, 
            batch_size=256, shuffle=False)    
        #test(net, loader_test)
        
        if method == 'BayesWRM' or method == 'Bayes':
            adversary = CWAttack(model_list, steps=steps, classes=43)
        else:
            adversary = CWAttack(net, steps=steps, classes=43)
        advacc = attack_over_test_data(net, adversary, param, loader_test)
        print('method',method,  'adv accuracy', advacc)
        advacc_set.append(advacc)

    df = pd.DataFrame(
        {'k': list(steps_set),
         'advacc': advacc_set})

    df.to_csv(os.path.join('trafficsignwhiteboxCW', method+'_FGSMwhitebox.csv'))


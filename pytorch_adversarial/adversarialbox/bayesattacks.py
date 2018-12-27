import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import torch
import torch.nn as nn
from adversarialbox.utils import to_var
from sampler.sghmc import SGHMC
from sampler.sgadahmc import SGAdaHMC

from torch.autograd import Variable
import os
import cv2

class ZNN(nn.Module):
    def __init__(self, zinput, y_var):
        super(ZNN, self).__init__()
        self.zinput = zinput
        self.y_var = y_var
        self.Z_var = nn.Parameter(torch.from_numpy(zinput.astype(np.float32)).cuda(), requires_grad=True)
        self.loss_fn = nn.CrossEntropyLoss()
                    
    def forward(self, model_list, gamma, Stheta):
                    
        Z_var0 = Variable(torch.from_numpy(self.zinput.astype(np.float32)).cuda(), requires_grad=False)
        total_loss = 0
        for i in range(Stheta):
            scores = model_list[i](self.Z_var)
            #loss = self.loss_fn(scores, y_var) - self.gamma*torch.sum((self.Z_var-Z_var0)**2)
            #Tracer()()
            loss = self.loss_fn(scores, self.y_var) - gamma*torch.sum((self.Z_var-Z_var0)**2)
            total_loss = total_loss + loss
        #return the negative loss to be maximized
        #total_loss = -total_loss/Stheta
        total_loss = -total_loss
        return total_loss

class BayesWRM(object):
    def __init__(self, model=None, model_list=None, Stheta=10, Sz=10, epsilon=0.5, k=15, a=1, gamma=2, T=1e-3, optimizer='SGHMC', random_start=False, clip=False, proj='infinorm', storeadv=False):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        Bayes WRM
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a #adv learning rate
        self.rand = random_start
        self.gamma = gamma
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.T = T #Temperature
        self.model_list = model_list
        
        self.Stheta = Stheta
        self.Sz = Sz
        self.clip = clip
        self.proj = proj
        self.optim = optimizer


        self.storeadv = storeadv

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        import time
        start_time = time.time()
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)
        Sz = self.Sz
        Stheta = self.Stheta
        MC = 15
        #List of numpy arrays
        z_list = [copy.deepcopy(X) for i in range(Sz)]
        y_list = [copy.deepcopy(y) for i in range(Sz)]

        y = np.concatenate(y_list)
        #print('X.shape', X.shape) 
        z_stack = np.concatenate(z_list)
         
        #print('Z_stack shape', z_stack.shape)
        
        y_var = to_var(torch.LongTensor(y))
        loss_fn = nn.CrossEntropyLoss()
            
                
        znn = ZNN(zinput=z_stack, y_var=y_var)         
        #optimizer = torch.optim.SGD(znn.parameters(), lr=1e-4)  
        
        if self.optim == 'SGHMC':
            optimizer = SGHMC(znn.parameters(), config=dict(lr=self.a, T=self.T, L=self.k))
        elif self.optim == 'SGAdaHMC':
            optimizer = SGAdaHMC(znn.parameters(), config=dict(lr=self.a, T=self.T, L=self.k))
            
        def helper():
            def feval():
                total_loss = znn(model_list=self.model_list, gamma=self.gamma, Stheta=Stheta)
                optimizer.zero_grad()
                total_loss.backward()
                return total_loss #TODO return loss for extension
            return feval
        total_loss = optimizer.step(helper())
        z_stack = znn.Z_var.data.cpu().numpy()

        #print('z_stack.shape', z_stack.shape)
        
        #print('inner maximization time %s' % (time.time()-start_time))
        #print('Sz step finished')
        
        z_list = []
        for i in range(Sz):
            batch_size = int(z_stack.shape[0]/Sz)
            #z_list.append(z_stack[i*batch_size:(i+1)*batch_size, :, :, :])
            z_adv = z_stack[i*batch_size:(i+1)*batch_size, :, :, :]

            z_list.append(z_adv)

            #print('maximum diff adv', np.max(np.abs(z_list[-1]-X_nat)))

        #print('Sz', Sz)
        #print('Stheta', Stheta)
        #print('len(z_list)', len(z_list))
        if self.storeadv == True:
            for ind, X in enumerate(z_list):
                X = np.clip(X, 0, 1)*255
                cv2.imwrite(os.path.join('advtrain_sample', 'BayesWRM_Advtraining_'+str(ind)+'.png'), np.squeeze(X[0,:]))
            exit(0)
        
        return z_list
        

import torch
from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
'''
SGHMC math similar to the Bayesian GAN
use lr = lr/sqrt(t) learning rate decay 

ref to the Entropy SGD implementation
'''

class SGHMC(Optimizer):
    def __init__(self, params, config):
        defaults = dict(lr=0.5, alpha=0, gamma=0.01, L=15, T=1e-5)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]
        
        super(SGHMC, self).__init__(params, defaults)
        self.config = config
    
    def __setstate__(self, state):
        super(SGHMC, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
          
        #parameter setting
        c = self.config
        L = int(c['L']) #number of Langevin steps
        gamma = c['gamma'] # 0.01
        lr = c['lr'] #lr=0.5
        T = c['T'] #1e-5
        #inner sampling steps
        for i in range(L):
            lr = float(lr/np.sqrt(L+1))
            f = closure()
            
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    d_param = param.grad.data
                    
                    param_state = self.state[param]
                    if 'mometum' not in param_state:
                        momentum = param_state['momentum']=torch.zeros(param.data.size()).cuda()
                        noise = deepcopy(momentum)
                    
                    noise.normal_()
                    momentum = (1-gamma)*momentum - lr*d_param + float(np.sqrt(2*lr*gamma*T))*noise
                    param.data.add_(momentum)    

        return loss

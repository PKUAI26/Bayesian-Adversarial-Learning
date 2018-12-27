import copy
import numpy as np
import cv2

from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch.autograd import Variable

from adversarialbox.utils import to_var, make_one_hot
import torch_extras
import os

advtrainfolder = 'advtrain_sample'





class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None, order='inf', is_train=True, storeadv=False, storeindex=0, advtraining='FGSM', pixelattack=0):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()
        self.order = order
        self.is_train = is_train


        #store intermediate results
        self.storeadv = storeadv
        self.storeindex = storeindex

        #if pixel attack
        self.pixelattack = pixelattack
        if pixelattack != 0:
            print('use the pixel attack')

        #advtraining method
        self.advtraining = advtraining
        
        

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)
        
        if type(self.model) is list:
            grad_list = []
            for model in self.model:
                X_var = to_var(torch.from_numpy(X), requires_grad=True)

                y_var = to_var(torch.LongTensor(y))
                scores = model(X_var)
                loss = self.loss_fn(scores, y_var)
                loss.backward()
                grad_list.append(X_var.grad.data.cpu().numpy())
            #grad_sign = np.mean(grad_list).sign().numpy()
            grad_sign = np.sign(np.mean(grad_list, axis=0))
            X += self.epsilon * grad_sign
            X = np.clip(X, 0, 1)
            #print('maximum diff adv', np.max(np.abs(X-X_nat)))
        else:
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))
         
            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            
            if self.pixelattack == 0:
                if self.order is 'inf':
                    grad_sign = X_var.grad.data.cpu().sign().numpy()
                    normalized_grad = grad_sign

                elif self.order is '2':
                    grad = X_var.grad.data.cpu().numpy()
                    square = sum(grad**2)
                    normalized_grad = grad / np.sqrt(square)
            
            elif self.pixelattack != 0:
                grad = X_var.grad.data.cpu().numpy()
                topk = grad.flatten()
                topk.sort()
                topk = topk[-self.pixelattack]
                grad[grad<topk] = 0.0
                grad[grad>=topk] = 1.0
                normalized_grad = grad

            else:
                raise NotImplementedError('Only L-inf, L2 norms FGSM attacks are implemented')

            X += self.epsilon * normalized_grad
            
            if self.is_train == False:
                X = np.clip(X, 0, 1)
            #print('maximum diff adv', np.max(np.abs(X-X_nat)))
        
        if self.storeadv == True:
            X_display = np.clip(X, 0, 1)*255
            cv2.imwrite(os.path.join(advtrainfolder, self.advtraining+'_epsilon_'+str(self.epsilon)+'_fixedindex_'+str(self.storeindex)+'_FGSMAttack.png'), np.squeeze(X_display[0,:]))
            #exit(0)

        return X

class PGDAttack(object):
    def __init__(self, model=None, epsilon=0.05, k=15, order='2', is_train=True, storeadv=False):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.loss_fn = nn.CrossEntropyLoss()
        self.order = order

        self.is_train = is_train

        self.storeadv = storeadv
        print('Init PGD, order:', order)

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            
            if self.order is 'inf':
                X += self.epsilon * np.sign(grad)
            elif self.order is '2':
                #print('grad shape', grad.shape)
                #square = sum(grad**2)
                grad = grad**2
                square = np.sum(grad, axis=2)
                square = np.sum(square, axis=2)
                square = np.squeeze(square)
                normalized_grad = (grad.T / (np.sqrt(square))).T
                X += self.epsilon * normalized_grad
            else:
                raise NotImplementedError('Only L-inf, L2 norms FGSM attacks are implemented')
                
            #X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
        if self.is_train == False:
            X = np.clip(X, 0, 1) # ensure valid pixel range
        #print('maximum diff adv', np.max(np.abs(X-X_nat)))
        if self.storeadv == True:
            X = np.clip(X, 0, 1)*255
            print('self.k', self.k, 'self.order', self.order)

            if self.k == 1 and self.order is '2':
                cv2.imwrite(os.path.join(advtrainfolder, 'PGD_Advtraining.png'), np.squeeze(X[0,:]))
            else:
                cv2.imwrite(os.path.join(advtrainfolder, 'IFGSM_Advtraining.png'), np.squeeze(X[0,:]))

            exit(0)
        
        return X

class CWAttack(object):
    def __init__(self, model=None, eps=0.3, steps=1, classes=10, storeadv=False, advtraining='FGSM'):
        #A simplified version of CW Attack
        #Running WRM requires batch size to be 1
        self.model = model
        self.eps = eps
        self.steps = steps
        self.loss_fn = nn.CrossEntropyLoss()

        self.storeadv = storeadv

        self.advtraining = advtraining
        self.classes = classes

    def perturb(self, X_nat, y):
        #refer to TF clevans implementation
        X = np.copy(X_nat)
        #X = np.clip(X, 0, 1)
        #X = 2*X - 1
        #X = np.arctanh(X*.9999)
        batch_size = X.shape[0]

        index = y.view(-1,1)

        #print('X.shape', X.shape)
        tlab = torch_extras.one_hot((X.shape[0], self.classes), index)
        tlab = to_var(torch.from_numpy(tlab.numpy().astype(np.long)), requires_grad=False)

        
        lower_bound = np.zeros(batch_size)

        X_var0 = to_var(torch.from_numpy(X), requires_grad=False)

        X_adv = np.copy(X_nat)

        if type(self.model) is list:
            for t in range(self.steps):
                grad_list = []
                for model in self.model:
                    X_var = to_var(torch.from_numpy(X), requires_grad=True)
                    y_var = to_var(torch.LongTensor(y))

                    scores = model(X_var)
                    tlab = tlab.type(torch.cuda.FloatTensor) 
                    real = torch.sum(torch.mul(scores, tlab))
                    other = torch.sum(torch.mul(scores, 1-tlab))
                    
                    loss1 = torch.clamp(real-other,min=0.0)
                    #loss1 = real-other
                    loss2 = (torch.sum((X_var-X_var0)**2)+1e-9)**0.5
                    
                    loss = loss1 + loss2

                    loss.backward()
                    grad = X_var.grad.data.cpu().numpy()
                    grad_list.append(grad)

                grad = np.mean(grad_list, axis=0)

                #go to the oposite direction as we wants to minimize the scores of the true label and maximize the scores of the wrong label
                X_adv = X_adv - 1./np.sqrt(t+2)*grad

        else:
            for t in range(self.steps):
                X_var = to_var(torch.from_numpy(X), requires_grad=True)
                y_var = to_var(torch.LongTensor(y))

                scores = self.model(X_var)
                tlab = tlab.type(torch.cuda.FloatTensor) 
                real = torch.sum(torch.mul(scores, tlab))
                other = torch.sum(torch.mul(scores, 1-tlab))
                
                loss1 = torch.clamp(real-other,min=0.0)
                #loss1 = real-other
                loss2 = (torch.sum((X_var-X_var0)**2)+1e-9)**0.5
                
                loss = loss1 + loss2

                loss.backward()
                grad = X_var.grad.data.cpu().numpy()   
                

                X_adv = X_adv - 1./np.sqrt(t+2)*grad

        #print('maximum diff adv', np.max(np.abs(X_adv-X_nat)))
        if self.storeadv == True:
            X = np.clip(X, 0, 1)*255
            cv2.imwrite(os.path.join(advtrainfolder, self.advtraining+'CWattacked.png'), np.squeeze(X[0,:]))
            exit(0)
        X_adv = np.clip(X_adv, 0, 1)
        return X_adv

    # --- Black-box attacks ---

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev)+ind] = X_sub[ind] + lmbda * grad_val #???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub

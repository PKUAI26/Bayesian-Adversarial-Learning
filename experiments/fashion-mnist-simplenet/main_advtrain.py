# coding: utf-8
# Try to merge simple advtrain and bayes advtrain
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

from adversarialbox.attacks import FGSMAttack, CertAttack, WRMAttack, PGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from adversarialbox.bayesattacks import BayesWRM
from sampler.sghmc import SGHMC
from sampler.sgadahmc import SGAdaHMC


#store xp values
#xp = []

#define adversary
parser = argparse.ArgumentParser()
#parser.add_argument("--model", type=str, default='FashionSimpleNet', help="model")
parser.add_argument("--advtraining", type=str, default='BayesWRM', help="adv training type")
parser.add_argument("--epsilon", type=float, default=0.02, help="adv training epsilon")
parser.add_argument("--Epoch", type=int, default=2, help="adv training epsilon")
#WRM parameters
parser.add_argument("--eps", type=float, default=0.1, help="adv training epsilon")
parser.add_argument("--storeadv", type=bool, default=False, help="whether store")
#BayesWRM and BNN parameters
parser.add_argument("--a", type=float, default=460, help="adv learning rate for BayesWRM")
parser.add_argument("--T", type=float, default=0.01, help="adv Temperature for inner maximization")
parser.add_argument("--initlr", type=float, default=0.01, help="Temperature for out maximization")
parser.add_argument("--T_out", type=float, default=1e-5, help="Temperature for out maximization")
parser.add_argument("--L", type=int, default=15, help="Number of iterations for solving maximization")
parser.add_argument("--Sz", type=int, default=5, help="Number of iterations for solving maximization")
parser.add_argument("--Stheta", type=int, default=5, help="Number of iterations for solving maximization")
parser.add_argument("--gamma", type=float, default=0.05, help="regularization param")
parser.add_argument("--multi", type=bool, default=False, help="Whether use multi method")
parser.add_argument("--inneroptimizer", type=str, default='SGHMC', help="optimizer to sample")
parser.add_argument("--outoptimizer", type=str, default='SGAdaHMC', help="optimizer to sample")
parser.add_argument("--batchsize", type=int, default=256, help="batchsize in training (specific for WRM)")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()



torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

param = {
        'model'     :   'SimpleNet',
        'patience'  :   args.Epoch,
        'batch_size':   args.batchsize,
        'nepochs'   :   args.Epoch,
        'nworkers'  :   1,
        'seed'      :   1,
        'data'      :   'fashion',
        'epsilon'   :   args.epsilon,
        }


advtraining = args.advtraining
print('======================================')

if advtraining == 'FGSM':
    adversary = FGSMAttack(epsilon=param['epsilon'])
    T = 0.0
    print('use FGSM adv training')
elif advtraining == 'IFGSM':
    #adversary = LinfPGDAttack(epsilon=param['epsilon'], k=15,order='inf')
    adversary = PGDAttack(epsilon=param['epsilon'], k=15,order='inf', storeadv=args.storeadv)
    T = 0.0
    print('use LinfPGD adv training')

elif advtraining == 'PGD':
    #adversary = LinfPGDAttack(epsilon=param['epsilon'], k=1, order='2')
    adversary = PGDAttack(epsilon=param['epsilon'], k=1, order='2', storeadv=args.storeadv)
    T = 0.0
    print('use PGD advtraining')

elif advtraining == 'WRM2':
    adversary = WRMAttack(eps=args.eps)
    T = 0.0
    print('use WRM2 adv training')

elif advtraining == 'BayesWRM':
    Sz = args.Sz
    Stheta = args.Stheta
    adversary = BayesWRM(a=args.a, k=args.L, epsilon=args.epsilon, gamma=args.gamma, T=args.T, Sz=Sz, Stheta=Stheta)
elif advtraining == 'Bayes':
    Sz = args.Sz
    Stheta = args.Stheta
    adversary = None


elif advtraining == 'ERM':
    adversary = None
    T = 0.0
    param['nepochs'] = args.Epoch = 15
    param['npatience'] = 15
    print('use ERM training')
else:
    raise NotImplementedError("Adv attack type note implemented yet")


#uda = not args.nocuda and torch.cuda.is_available() # use cuda
cuda = True
print('Training on cuda: {}'.format(cuda))

'''
# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(param['seed'])
if cuda:
    torch.cuda.manual_seed(param['seed'])
'''
# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup tensorboard folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(param['model'])
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
	run += 1
	current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
#print(args, file=logfile)

# Tensorboard viz. tensorboard --logdir {yourlogdir}. Requires tensorflow.
#from tensorboard_logger import configure, log_value
#configure(current_dir, flush_secs=5)

# Define transforms.
# normalize = transforms.Normalize((0.1307,), (0.3081,)
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

# Create dataloaders. Use pin memory if cuda.
kwargs = {'pin_memory': True} if cuda else {}
if(param['data'] == 'mnist'):
    trainset = datasets.MNIST('data-mnist', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(trainset, batch_size=param['batch_size'],
                            shuffle=True, num_workers=args.nworkers, **kwargs)
    valset = datasets.MNIST('data-mnist', train=False, transform=val_transforms)
    val_loader = DataLoader(valset, batch_size=param['batch_size'],
                            shuffle=False, num_workers=param['nworkers'], **kwargs)
else:
    trainset = FashionMNIST('data', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(trainset, batch_size=param['batch_size'],
                            shuffle=True, num_workers=param['nworkers'], **kwargs)
    valset = FashionMNIST('data', train=False, transform=val_transforms)
    val_loader = DataLoader(valset, batch_size=param['batch_size'],
                            shuffle=False, num_workers=param['nworkers'], **kwargs)







net = model.__dict__[param['model']]()

if advtraining != 'ERM':
    net.load_state_dict(torch.load('ERM.model'))
    #net.load_state_dict(torch.load('./advtrained_models/WRM2.model'))

criterion = torch.nn.CrossEntropyLoss()
if cuda:
    net, criterion = net.cuda(), criterion.cuda()



import copy
if advtraining == 'BayesWRM' or advtraining == 'Bayes':
    model_list=[copy.deepcopy(net) for i in range(Stheta)]
    for net in model_list:
        net.train()
    optimizer_list = [SGAdaHMC(model_list[i].parameters(), config=dict(lr=args.initlr, T=args.T_out)) for i in range(args.Stheta)]
#print(net)

# early stopping parameters
patience = param['patience']
best_loss = 1e4

# Print model to logfile
#print(net, file=logfile)

# Change optimizer for finetuning
if advtraining=='WRM2':
    optimizer = optim.Adam(net.parameters())
else:
    optimizer = optim.Adam(net.parameters(), lr=0.00001)



for e in range(param['nepochs']):
    print('Starting epoch %d' % (e+1))
    
    for t, (x_input, y_label) in enumerate(train_loader):
        #print('t:',t)
        x_var, y_var = to_var(x_input), to_var(y_label.long())

        if args.advtraining == 'BayesWRM' or args.advtraining == 'Bayes':
            if args.advtraining == 'BayesWRM':
                x_adv = x_input.cpu()
                x_adv = adv_train(X=x_adv, y=y_label.cpu().long(), model=model_list, criterion=criterion, adversary=adversary)   
            for i in range(len(model_list)):
                optimizer = SGAdaHMC(model_list[i].parameters(), config=dict(lr=args.initlr, T=args.T_out))
                #optimizer = optimizer_list[i]
                if advtraining == 'BayesWRM':
                    def helper():
                        def feval():
                            loss_adv = 0
                            for k in range(len(x_adv)):
                                x_adv_var = to_var(torch.from_numpy(x_adv[k].astype(np.float32)))
                                #loss_adv = loss_adv + criterion(net(x_adv_var), y_var)
                                loss_adv = loss_adv + criterion(model_list[i](x_adv_var), y_var)
                                loss = criterion(model_list[i](x_var), y_var)
                                loss_adv = loss_adv + loss
                            loss_adv = loss_adv/2.0
                            #loss_adv = loss_adv/len(x_adv)
                            optimizer.zero_grad()
                            loss_adv.backward()
                            
                            return loss_adv #TODO return loss for extension
                        return feval
                elif advtraining=='Bayes':
                    def helper():
                        def feval():
                            loss =criterion(model_list[i](x_var), y_var)
                            optimizer.zero_grad()
                            loss.backward()
                        
                            return loss #TODO return loss for extension
                        return feval
                else:
                    raise NotImplementedError('advtraining method not implemented')
                loss_total = optimizer.step(helper())
                if t%100 == 0:
                    print('loss', loss_total)
        else:
            #simple adv training method
            X, y= x_input.cuda(),y_label.cuda() 
            if adversary is not None:
                x_adv = X.cpu()

                x_adv = adv_train(x_adv, y.cpu().long(), net, criterion, adversary)
            
                x_adv_var = to_var(x_adv)
                loss_adv = criterion(net(x_adv_var), y_var)
                loss = (loss_adv + criterion(net(x_var), y_var))/2.0
            #ERM training
            else:
                x_var, y_var = to_var(X), to_var(y.long())
                loss = criterion(net(x_var), y_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




    # print stats
    #log_value('train_loss', train_loss, e)
    #log_value('val_loss', val_loss, e)
    #log_value('train_acc', train_acc, e)
    #log_value('val_acc', val_acc, e)
import os

if advtraining == 'ERM':
    #Save ERM trained model
    torch.save(net.state_dict(), 'ERM.model')
elif advtraining == 'BayesWRM' or advtraining == 'Bayes':

    for ind in range(args.Stheta):
        torch.save(model_list[ind].state_dict(), os.path.join('advtrained_models', advtraining+str(ind)+'.model'))
else:
    torch.save(net.state_dict(), os.path.join('advtrained_models', advtraining+'.model'))


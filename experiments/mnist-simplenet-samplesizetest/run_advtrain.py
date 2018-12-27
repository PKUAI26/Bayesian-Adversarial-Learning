import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=5, help="Number of samples")
args = parser.parse_args()

method_set = ['FGSM' ,'IFGSM', 'PGD', 'ERM', 'WRM2', 'BayesWRM', 'Bayes']

a = 300
T = 1e-5
Sz = 1
Stheta = args.sample
initlr=0.015
Tout=1e-5
rounds = 1
gamma=1e-4

for rind in range(rounds):
    for method in method_set:
        if method == 'BayesWRM':
            os.system('python main_advtrain.py --advtraining=BayesWRM --gamma='+str(gamma)+' --Sz='+str(Sz)+' --Stheta='+str(Stheta)+' --a='+str(a)+' --T='+str(T)+' --initlr='+str(initlr)+' --T_out='+str(Tout)) 
        elif method == 'Bayes':
            os.system('python main_advtrain.py --advtraining=Bayes '+ ' --Sz='+str(Sz)+' --Stheta='+str(Stheta)+' --a='+str(a)+' --T='+str(T)+' --initlr='+str(initlr)+' --T_out='+str(Tout)) 
        elif method == 'WRM2':
            os.system('python main_advtrain.py  --eps=5e1 --advtraining='+method) 
        elif method == 'FGSM':
            os.system('python main_advtrain.py  --epsilon=0.02 --advtraining='+method) 
        elif method == 'IFGSM':
            os.system('python main_advtrain.py  --epsilon=0.001 --advtraining='+method) 
        elif method =='PGD':
            os.system('python main_advtrain.py  --epsilon=4e2 --advtraining='+method) 
        else:
            raise NotImplementedError('Not implemented')


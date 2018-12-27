import os


method_set = ['Bayes', 'BayesWRM', 'FGSM', 'IFGSM', 'PGD']

a = 500
T = 1e-5
#initlr=0.001
initlr=0.01
T_out = 1e-5
Epoch = 1
for rind in range(1):
    for method in method_set:
        if method == 'PGD':
            os.system('python traffic_sign_advtrain.py --epsilon=2e3 --advtraining='+method)
        elif method == 'FGSM':
            os.system('python traffic_sign_advtrain.py --epsilon=0.0345 --advtraining='+method)
        elif method == 'IFGSM':
            os.system('python traffic_sign_advtrain.py --epsilon=0.0023 --advtraining='+method)
        elif method == 'BayesWRM':
            os.system('python traffic_sign_advtrain.py '+'--a='+str(a)+' --T='+str(T)+' --initlr='+str(initlr)+' --T_out='+str(T_out)+' --Epoch='+str(Epoch)) 
        elif method == 'Bayes':
            os.system('python traffic_sign_advtrain.py --advtraining=Bayes '+'--a='+str(a)+' --T='+str(T)+' --initlr='+str(initlr)+' --T_out='+str(T_out)+' --Epoch='+str(Epoch)) 
        else:
            os.system('python traffic_sign_advtrain.py --advtraining='+method)



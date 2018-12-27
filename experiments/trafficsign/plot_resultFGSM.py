import os
import pandas as pd
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import argparse

font = {'family' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
parser = argparse.ArgumentParser()
parser.add_argument("--blackbox", type=bool, default=False, help="whether use blackbox")
parser.add_argument("--save", type=bool, default=True, help="whether save fig")
args = parser.parse_args()

rounds = 1
if args.blackbox == False:
    problem = 'trafficsignwhitebox'
else:
    problem = 'trafficsignblackbox'
method = ['BayesWRM', 'Bayes', 'FGSM', 'IFGSM', 'PGD', 'WRM2', 'ERM']
#Note that in this code, the name of the method should have different prefix

'''
if 'blackbox' in  problem:
    method = ['BayesWRM', 'Bayes', 'FGSM', 'IFGSM', 'PGD', 'WRM2']
'''

'''
if problem == 'fashionmnistblackbox':
    method = ['BayesWRM', 'FGSM', 'IFGSM']
'''
plt.figure()
#plt.yscale('log')
for med in method:

    print(med)
    advacc_set = []
    for rind in range(1, rounds+1):

        #print('round', rind, 'method', med)
        resultfile = glob.glob(os.path.join(problem,  med+'_FGSM*'))[0]
        #print(resultfile)

        df = pd.read_csv(resultfile)

        advacc = df['advacc'].as_matrix()
        epsilon = df['epsilon'].as_matrix()

        #advacc = 1 - advacc
        advacc_set.append(advacc)

    advacc_set = np.stack(advacc_set)
    #print('advacc_set shape', advacc_set.shape)
    advacc_mean = np.mean(advacc_set, axis=0)
    advacc_std = np.std(advacc_set, axis=0)

    #print(advacc_std)
    #exit(0)

    plt.plot(epsilon, advacc_mean)
    #plt.errorbar(epsilon, advacc_mean, yerr=advacc_std)
    print('Method', med, 'epsilon=0, mean acc:', advacc_mean[0])

if problem == 'trafficsignwhitebox':
    plt.ylim([0, 1])
    #plt.axhline(y=1/43.0, color='k', linestyle='--')
    #plt.axvline(x=0.08, color='k', linestyle='--')
    plt.xlim([0, 0.1])
elif problem == 'trafficsignblackbox':
    plt.ylim([0, 1])
    #plt.axhline(y=1/43.0, color='k', linestyle='--')
    #plt.axvline(x=0.08, color='k', linestyle='--')
    plt.xlim([0, 0.1])

elif problem == 'fashionmnistwhitebox' or problem == 'fashionmnistblackbox':
    plt.ylim([0.05, 1])
    plt.axhline(y=1/10.0, color='k', linestyle='--')
    



plt.legend(['BAL', 'BNN', 'FGSM', 'IFGSM', 'PGD', 'WRM', 'ERM'])

'''
if 'blackbox' not in problem:
    plt.legend(['BAL', 'BNN', 'FGSM', 'IFGSM', 'PGD', 'WRM', 'ERM'])
else:
    plt.legend(['BAL', 'BNN', 'FGSM', 'IFGSM', 'PGD', 'WRM'])
'''
plt.xlabel(r'FGSM attack $\epsilon$')
plt.ylabel('Accuracy')
plt.ylim([0, 0.4])
if args.save == True:
    plt.savefig(problem+'result.pdf', dpi=400)
plt.show()

    




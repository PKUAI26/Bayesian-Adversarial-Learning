import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import argparse

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument("--blackbox", type=bool, default=False, help="whether use blackbox")
parser.add_argument("--save", type=bool, default=True, help="whether to save fig")
args = parser.parse_args()


problem = 'mnistwhiteboxCW'

method = ['BayesWRM', 'Bayes', 'FGSM', 'IFGSM', 'PGD', 'WRM2', 'ERM']


advacc_set = []
for med in method:
    
    resultfile = glob.glob(os.path.join(problem,  med+'_*'))[0]
    print(resultfile)

    df = pd.read_csv(resultfile)

    advacc = df['advacc'].as_matrix()[0]

    advacc_set.append(advacc)


font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
plt.figure()

method = ['BAL', 'BNN', 'FGSM', 'IFGSM', 'PGD', 'WRM', 'ERM']
plt.bar(range(len(advacc_set)), advacc_set)
plt.xticks(range(len(advacc_set)), method)
plt.ylabel('Accuracy ')
plt.ylim([0, 1])
plt.xlabel('Carlini-Wagner attack result')
plt.tight_layout()
plt.savefig('mnistwhiteboxCWresult.pdf', format='pdf')
plt.show()


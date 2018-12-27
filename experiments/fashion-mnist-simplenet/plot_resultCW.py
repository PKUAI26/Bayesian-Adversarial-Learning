import os
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
plt.figure()
means = [0.0981, 0.0833, 0.0806, 0.0813, 0.0604, 0.1458,0.0679]
method = ['BAL', 'BNN', 'FGSM', 'IFGSM', 'PGD', 'WRM', 'ERM']
plt.bar(range(len(means)), means)
plt.xticks(range(len(means)), method)
plt.ylabel('Accuracy ')
plt.ylim([0, 1])
plt.xlabel('Carlini-Wagner attack result')
plt.savefig('fashionmnistCW.pdf', format='pdf')
plt.show()

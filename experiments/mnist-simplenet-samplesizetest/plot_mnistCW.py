import os
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
plt.figure()
means = [0.2886, 0.3455, 0.3512, 0.4322]
method = ['1', '3', '5', '10']
plt.bar(range(len(means)), means)
plt.xticks(range(len(means)), method)
plt.ylabel('Accuracy ')
plt.ylim([0, 1])
plt.xlabel('Carlini-Wagner attack result')
plt.savefig('mnistCW.pdf', format='pdf')
plt.show()

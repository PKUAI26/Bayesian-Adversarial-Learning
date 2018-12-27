import os

samples=[1,  3,  5, 10]
for sample in samples:
    os.system('python run_advtrain.py --sample='+str(sample))
    os.system('python run_advattackFGSM.py --sample='+str(sample))
    os.system('python run_advattackCW.py --sample='+str(sample))

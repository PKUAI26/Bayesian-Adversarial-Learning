
# Bayesian Adversarial Learning 
## Introduction 
We propose a novel framework for Bayesian  adversarial learning that can be applied to various applications such as adversarial defense. 

## Prerequisites 
Our code is based on Python3 (>=3.5). There are a few dependencies to run the code. Please stick to the versions listed.
The major libraries are listed as follows:
* PyTorch (= 0.4.0)
* Cuda Toolkit (= 9)


## Dataset
### Dataset need to download before running
**[Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads)**  Note that we only use the training set from the original dataset and split it into training and test in experiments.
We provide a processed dataset which can be obtained from https://drive.google.com/open?id=1gPreM_0RWMCA0qwzZpoyWWy3g5FMGGYj
Note that please cite the original dataset and meet the requirements.


## How to quickly reproduce the results


cd bayesian_adversarial_learning_release/experiments/trafficsign

To test on FGSM Attack:

python run_advattackFGSM.py

python plot_resultFGSM.py

To test on CW Attack:

python run_advattackCW.py






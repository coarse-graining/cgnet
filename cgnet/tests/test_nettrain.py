# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import numpy as np
import json
import tempfile
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from cgnet.network import CGnet, LinearLayer, ForceLoss
from cgnet.train import Trainer
from cgnet.feature import MoleculeDataset

# Random train data
x0 = np.random.randn(10,2).astype('float32')
y0 = np.random.randn(10,2).astype('float32')

# Random test data
x0_test = np.random.randn(10,2).astype('float32')
y0_test = np.random.randn(10,2).astype('float32')


# Placeholder dataset 
train_set = MoleculeDataset(x0,y0)
test_set = MoleculeDataset(x0_test,y0_test)

# Toy model
arch = LinearLayer(2,10) +\
       LinearLayer(10,10) +\
       LinearLayer(10,1)

batch_sizes = [32,64,128,256,512,1024]

train_bs = batch_sizes[np.random.randint(4)]
test_bs = batch_sizes[np.random.randint(4)]
train_sampler = RandomSampler(train_set)
test_sampler = SequentialSampler(test_set)

model = CGnet(arch,ForceLoss()).float()
optimizer = Adam(model.parameters(),lr=0.001)
trainloader = DataLoader(train_set, batch_size=64, sampler=train_sampler)
testloader = DataLoader(test_set, batch_size=64, sampler=test_sampler)

def test_train_class():
    """Test Trainer class"""

    with tempfile.TemporaryDirectory() as tmp:
        num_epochs = 10#np.random.randint(1,high=10)
        gam = np.random.uniform(low=0.1,high=0.9)
        scheduler = MultiStepLR(optimizer,milestones = [1,2,3,4,5],\
                                             gamma =  gam)

        trainer = Trainer(trainloader=trainloader,testloader=testloader,
                          optimizer=optimizer,scheduler=scheduler,log=True,
                          save_dir=tmp+"/",save_freq=5)

        trainer.train(model,num_epochs,verbose=False)

        np.testing.assert_equal(num_epochs,len(trainer.epochal_train_losses))
        np.testing.assert_equal(num_epochs,len(trainer.epochal_test_losses))








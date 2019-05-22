import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from network.nnet import Net, LinearLayer, ForceLoss
from train.nettrain import Trainer

# Random train data
x0 = torch.rand((3000,2),requires_grad=True)
y0 = torch.rand((3000,2))
batch = {'traj':x0, 'force':y0}

# Random test data
x0_test = torch.rand((3000,2),requires_grad=True)
y0_test = torch.rand((3000,2))

# Placeholder dataset 
train_set = TensorDataset(x0,y0)
test_set = TensorDataset(x0_test,y0_test)

# Toy model
arch = LinearLayer(2,10) +\
       3*LinearLayer(10,10) +\
       LinearLayer(10,2)

batch_sizes = [32,64,128,256,512,1024]

train_bs = batch_sizes[np.random.randint(4)]
test_bs = batch_sizes[np.random.randint(4)]

model = Net(arch,ForceLoss)
optimizer = Adam(model.parameters(),lr=0.001)
trainLoader = DataLoader(train_set,batch_size=train_bs,sampler=RandomSampler())
testloader  = DataLoader(test_set,batch_size_test_bs,sampler=SequentialSampler())

def test_train_class():
    """Test Trainer class"""
    num_epoch = np.random.randint(10)
    gam = np.random.randn(1.0)
    scheduler = lr_scheduler.MultiStepLR(milestones = np.arange(1,num_epoch,1),\
                                         gamma =  gam)

    trainer = Trainer(trainloader=trainloader,testloader=testloader,
                      optimizer=optimizer,scheduler=scheduler)
    trainer.train(model,num_epochs,verbose=False)
    np.testing.assert_equal(num_epochs,len(trainer.epochal_train_losses)
    np.testing.assert_equal(num_epochs,len(trainer.epochal_test_losses)

    # Test to see if losses are being recorded properly

    data_loss = trainer.dataset_loss(model,testloader)

    # Test dataset_loss
    # Test gradient reset
    # Test Lipschitz constraint






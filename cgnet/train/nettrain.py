# Author: Nick Charron
# Contributors: Brooke Husic, Dominik Lemm

import torch
import torch.nn as nn
import time
import json

class Trainer():
    """Class or training networks"""

    def __init__(self, trainloader=None, testloader=None, optimizer=None,
                 scheduler=None, lipschitz=False, name='MyModel', save_dir=None,
                 log=False):
        """Initializaiton

        Parameters
        ----------
        trainloader : torch.data.loader() object (default=None)
            dataoder for training dataset
        testloader : torch.data.loader() object (default=None)
            dataloader for testing dataset
        optimizer : torch.optim.optimzer class (default=torch.optim.Adam())
            optimizer used for updating network weights
        scheduler : torch.optim.lr_scheduler object (default=None)
            learning rate scheduler
        lipschitz : bool or float (default=False)
            strength of L2 lipschitz projection after each
            optimizer step


        Attributes
        ----------
        epochal_train_losses : list
            losses recorded over the entire training set every epoch
        epochal_test_losses : list
            losses recorded over the entire test set every epoch

        """
        self.name = name
        self.save_dir = save_dir
        self.log = log
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.lipschitz = lipschitz
        self.scheduler = scheduler

        self.epochal_train_losses = []
        self.epochal_test_losses = []

    def make_log(self):
        """Produce JSON log of training and dump to file in the directory
        specified by self.save_dir

        Attributes
        ----------

        """

        self.log_data = {'training': {}, 'meta': {}}
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        if self.testloader:
            test_train_split = float(self.testloader.dataset.__len__())/self.trainloader.dataset.__len__()
        else:
            test_train_split = 0.00
        if self.scheduler:
            scheduler_dict = self.scheduler.state_dict()
        else:
            scheduler_dict = None
        self.log_data['training']['epochs'] = self.num_epochs
        self.log_data['training']['lr'] = lr
        self.log_data['training']['lipschitz'] = self.lipschitz
        self.log_data['training']['batch_size'] = self.trainloader.batch_size
        self.log_data['training']['split'] = test_train_split
        self.log_data['training']['scheduler'] = scheduler_dict
        self.log_data['training']['optimizer'] = self.optimizer.__class__.__name__

        self.log_data['meta']['date'] = self.date
        self.log_data['meta']['train_time'] = self.train_time
        self.log_data['meta']['save_dir'] = self.save_dir
        self.log_data['meta']['name'] = self.name

#        with open(self.save_dir + self.name + '.json') as log_file:
#            json.dump(self.log_data, log_file)

    def load_routine(self,routine):
        with open(routine) as log_file:
           self.log_data = json.load(routine)

    def dataset_loss(self, model, loader):
        """Compute average loss over arbitrary loader/dataset

        Parameters
        ----------
        model : Net() instance
            model to calculate loss
        loader : torch.utils.data.DataLoader() instance
            loader (with associated dataset)

        Returns
        -------
        loss : torch.Variable
            loss computed over the dataset

        """
        loss = 0
        num_batch = 0
        for num, batch in enumerate(loader):
            coords, force = batch
            loss += model.predict(coords, force)
            num_batch += 1
        loss /= num_batch
        return loss

    def lipschitz_projection(self, model):
        """Performs L2 Lipschitz Projection via spectral normalization

        Parameters
        ----------
        model : Net() instance
            model to perform Lipschitz projection upon

        """
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                u, s, v = torch.svd(weight)
                if next(model.parameters()).is_cuda:
                    lip_reg = torch.max(((s[0]) / self.lipschitz),
                                        torch.tensor([1.0]).double().cuda())
                else:
                    lip_reg = torch.max(((s[0]) / self.lipschitz),
                                        torch.tensor([1.0]).double())
                layer.weight.data = weight / (lip_reg)

    def train(self, model, num_epochs, verbose=True,
              batch_freq=1, epoch_freq=1):
        """Training loop

        Parameters
        ----------
        model : Net() instance
            model to train
        num_epochs : int
            number of epochs used for training (one epoch is defined
            as a single pass through the training dataset - ie, one epoch
            of training has finished after the optimizer has stepped over
            each batch in the training dataset)
        verbose : bool (default=True)
            if True, progress messages on training and validation error
            are reported to stdout for the specified frequency
        batch_freq : int (default=1)
            frequency of batches with which verbose messages are printed
        epoch_freq : int (default=1)
            frequency of epochs with which verbose messages are printed

        """
        self.num_epochs = num_epochs
        self.date = time.asctime(time.localtime(time.time()))
        start = time.time()
        for epoch in range(0,num_epochs):
            if self.scheduler:
                self.scheduler.step()
            test_loss = 0.00
            for num, batch in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                coords, force = batch
                energy, pred_force = model.forward(coords)
                batch_loss = model.criterion(pred_force, force)
                batch_loss.backward()
                self.optimizer.step()
                # perform L2 lipschitz check and projection
                if self.lipschitz:
                    self.lipschitz_projection(model)
                if verbose:
                    if num % batch_freq == 0:
                        print(
                            "Batch: {}\tTrain: {}\tTest: {}".format(
        num, batch_loss, test_loss))
            train_loss = self.dataset_loss(model, self.trainloader).data
            test_loss = self.dataset_loss(model, self.testloader).data
            if verbose:
                if epoch % epoch_feq == 0:
                    print(
                        "Epoch: {}\tTrain: {}\tTest: {}".format(
        epoch, train_loss, test_loss))
            self.epochal_train_losses.append(train_loss)
            self.epochal_test_losses.append(test_loss)
        end = time.time()
        self.train_time = end - start
        if self.log:
            self.make_log()

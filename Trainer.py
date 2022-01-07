"""@package Trainer

"""
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import torch.distributed as dist

class Trainer:
    """The trainer performs the main training loop when training. It takes
    a model, dataloader, loss function, and optimizer to train a model 
    using mini-batches. 

    There are also options for scheduler and early stopping.
    """
    def __init__(self, model, optimizer, loss_fn, device, dataloader_get_func, warmup_epochs=0, train_sampler=None, model_get_func=None, epochs=100, scheduler=None, epoch_load_bar=False, early_stop=True, distributed=False, early_stop_patience=10, checkpoint_frequency=5, plot=False, rank=0):
        """Initializes the instance of the Trainer to the training job.

        Args:
            model (torch.nn.Module): The model you wish to train.
            loss_fn (torch.nn.Module): Loss function to train with.
            optimizer (torch.optim): Optimizer which will perform backpropogation
            device (torch.device): Device that training will take place in.
            dataloader_get_func (method): Function that converts __getitem__ output from dataloader to input and labels tensors.
            model_get_func (method, optional): Method used to get predictions from model.
            epochs (int, optional): Number of epochs to train. Defaults to 100.
            scheduler (torch.optim.lr_scheduler, optional): Scheduler which changes learning rate of optimizer over time. Defaults to None.
            epoch_load_bar (bool, optional): Set to true to see load bar for each epoch. Defaults to False.
            early_stop (bool, optional): Set to True to stop training early if validation loss does not decrease in early_stop_patience epochs.
            early_stop_patience (int, optional): Number of epochs to allow non-decreasing validation loss before ending training
        """
        self.rank = rank
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.dataloader_get_func = dataloader_get_func
        self.warmup_epochs = warmup_epochs
        self.train_sampler = train_sampler
        self.model_get_func = model_get_func
        self.epochs = epochs
        self.scheduler = scheduler
        self.epoch_load_bar = epoch_load_bar
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.loss = {"train": [], "val": []}
        self.min_val_loss = float("inf")
        self.no_decrease_epochs = 0
        self.checkpoint_frequency = checkpoint_frequency
        self.plot = plot
        self.distributed = distributed
    
    def train(self, train_dataloader, val_dataloader):
        """[summary]

        Args:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training set.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation set.
        """
        self.model = self.model.to(self.device)
        for epoch in range(self.epochs):
            #Set epoch for sampler
            for sampler in self.train_sampler:
                sampler.set_epoch(epoch)

            # Perform an epoch of training
            self._epoch_train(train_dataloader,epoch)
            #Evaluate the model on the evaluation set
            self._epoch_eval(val_dataloader)

            #Print current state of the model
            self.__print_state(epoch)

            # If the scheduler is initialized, step it so it knows whether to decrease
            # the learning rate
            self.__step_scheduler(epoch)

            # If a checkpoint is reached, save the state dictionary
            self.__set_checkpoint(epoch)
            
            if self.plot:
                plt.plot(range(epoch+1),self.loss["train"], label="train")
                plt.plot(range(epoch+1), self.loss["val"], label="val")
                plt.show()
            
            #If early stop is set and model hasn't improved in early_stop_patience epochs,
            #stop the training
            if self.early_stop:
                if self.__early_stop():
                    print("Early Stopping")
                    torch.save(self.model.module.state_dict(), "model_final")
                    break




    def __print_state(self, curr_epoch):
        """Prints the current state of training. Shows the training loss, validation loss
        the current epoch, and the learning rate

        Args:
            curr_epoch (int): the current epoch
        """
        if self.rank == 0:
            print("Rank: {}, Epoch: {}/{}, Train Loss={}, Val Loss={}, LR={}".format(
                        self.rank,
                        curr_epoch+1,
                        self.epochs,
                        np.round(self.loss["train"][-1], 10),
                        np.round(self.loss["val"][-1], 10),
                        self.optimizer.param_groups[0]['lr']
            ))

    def _epoch_train(self, dataloader,epoch):
        """Train the model for a single epoch on the training datalaoder.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for training dataset
        """
        self.model.train()
        running_loss = []
        iterator = self.__get_iterator(dataloader, "Train set",epoch)

        for data in iterator:
            inputs, labels = self.dataloader_get_func(data)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(inputs)
            if type(outputs) == list:
                loss = 0
                for i in range(len(outputs)):
                    loss += self.loss_fn(outputs[i], labels)
                loss.backward()
            else:
                loss = self.loss_fn(outputs, labels)
                loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss.append(float(loss.item()))
        epoch_loss = np.mean(running_loss)
        if self.distributed:
            epoch_loss_tensor = torch.zeros(1)
            epoch_loss_tensor[0] = float(epoch_loss)
            epoch_sum_tensor = epoch_loss_tensor.to(self.device)
            dist.all_reduce(epoch_sum_tensor)
            epoch_loss = epoch_sum_tensor[0]
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader):
        """Evaluate model for a single epoch on the validation datalaoder.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for validation dataset
        """
        self.model.eval()
        running_loss = []
        iterator = self.__get_iterator(dataloader, "Validation set")
        with torch.no_grad():
            for data in iterator:
                inputs, labels = self.dataloader_get_func(data)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = 0
                if type(outputs) == list:
                    for i in range(len(outputs)):
                        loss += self.loss_fn(outputs[i], labels)
                else:
                    loss = self.loss_fn(outputs, labels)
                running_loss.append(float(loss.item()))
            epoch_sum = np.sum(running_loss)
            epoch_len = len(running_loss)
            epoch_loss = 0
            if self.distributed:
                epoch_sum_tensor = torch.zeros(1)
                epoch_sum_tensor[0] = float(epoch_sum)
                epoch_sum_tensor = epoch_sum_tensor.to(self.device)
                epoch_len_tensor = torch.zeros(1)
                epoch_len_tensor[0] = epoch_len
                epoch_len_tensor = epoch_len_tensor.to(self.device)
                dist.all_reduce(epoch_sum_tensor)
                dist.all_reduce(epoch_len_tensor)
                epoch_loss = epoch_sum_tensor[0]/epoch_len_tensor[0]
            else:
                epoch_loss = epoch_sum/epoch_len
            self.loss["val"].append(float(epoch_loss))


    def __get_iterator(self, dataloader, desc="",epoch=-1):
        """Get the appropriate iterator for the dataloader. The iterator may have a subset of the
        dataset batches. It also has the option for showing the loading bar

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to iterate through
            desc (str, optional): Loading bar description for tqdm. Defaults to "".

        Returns:
            Iterator: Iterator for the batches of the dataset
        """
        #If a the maximum batches per epoch has been set, then stop the batches at that point.
        #If loading bar is enabled, use tqdm with description
        iterator = tqdm(dataloader, leave=False, desc=desc) if self.epoch_load_bar else dataloader
        return iterator
    
    def __set_checkpoint(self, epoch):
        """Each checkpoint_frequency epochs, save the model state

        Args:
            epoch ([type]): [description]
        """
        if (epoch + 1) % self.checkpoint_frequency == 0 and self.rank == 0:
            now = datetime.now() # current date and time
            date_time = now.strftime("%d%m%Y%H%M%S")
            print("Saving Checkpoint")
            torch.save(
                self.model.module.state_dict(), "/data/model_{}_{}.pt".format(date_time,str(epoch+1).zfill(3))
            )

    def __step_scheduler(self, epoch):
        """If there is a scheduler, step it once with the latest validation loss.
        """
        if self.scheduler is not None:
            if type(self.scheduler) == list:
                if epoch < self.warmup_epochs:
                    self.scheduler[0].step()
                self.scheduler[1].step(self.loss["val"][-1])
            else:
                self.scheduler.step(self.loss["val"][-1])

    def __early_stop(self):
        """Determines if the model is no longer learning. If not, it will return true
        to stop and false to continue learning.

        Returns:
            bool: True if learning must stop, false otherwise.
        """
        val_loss = self.loss["val"][-1]
        if val_loss >= self.min_val_loss:
            self.no_decrease_epochs +=1
        else:
            self.min_val_loss = val_loss
            self.no_decrease_epochs = 0
        
        if self.no_decrease_epochs > self.early_stop_patience:
            return True
        else:
            return False 
import logging
import os
import asyncio
import numpy as np
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
sys.path.append("..")
from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):

    # 重写data loader，更改examples.to(device)的数据类型，loss函数中target数据类型的转变
    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        # logging.info("plato->trainers basic.py train_process()")

        if 'use_wandb' in config:
            import wandb

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 10
                batch_size = config['batch_size']

                # logging.info("[Client #%d] Loading the dataset.",
                #              self.client_id)
                _train_loader = getattr(self, "train_loader", None)

                if callable(_train_loader):
                    train_loader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    # train_loader = torch.utils.data.DataLoader(
                    #     dataset=trainset,
                    #     shuffle=False,
                    #     batch_size=batch_size,
                    #     sampler=sampler.get())

                    # 重新定义train loader
                    train_loader = DataLoader(dataset=trainset,
						batch_size=batch_size,
						shuffle=False)

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
                # logging.info("client @%d iterations_per_epoch: %d",self.client_id,iterations_per_epoch)
                # logging.info(trainset)
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

                # Initializing the loss criterion
                _loss_criterion = getattr(self, "loss_criterion", None)
                if callable(_loss_criterion):
                    loss_criterion = self.loss_criterion(self.model)
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                # Initializing the optimizer
                get_optimizer = getattr(self, "get_optimizer",
                                        optimizers.get_optimizer)
                optimizer = get_optimizer(self.model)

                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None


                for epoch in range(1, epochs + 1):
                    for batch_id, (examples,
                                   labels) in enumerate(train_loader):
                        # 改变数据类型
                        examples, labels = examples.to(device = self.device,dtype=torch.float), labels.to(
                            self.device)
                        if 'differential_privacy' in config and config[
                                'differential_privacy']:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            optimizer.zero_grad()

                        if cut_layer is None:
                            
                            outputs = self.model(examples)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)
                        # labels = torch.topk(labels, 1)[1].squeeze(1)

                        # 改变数据类型
                        loss = loss_criterion(outputs, labels.long())

                        loss.backward()
                        optimizer.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))

                    if lr_schedule is not None:
                        lr_schedule.step()

                    if hasattr(optimizer, "params_state_update"):
                        optimizer.params_state_update()

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()


    # 重写data loader，更改examples.to(device)的数据类型，loss函数中target数据类型的转变
    def test_process(self, config, testset, sampler=None):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        # logging.info("plato->trainers basic.py test_process()")

        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                if sampler is None:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False)
                # Use a testing set following the same distribution as the training set
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        sampler=sampler.get())

                correct = 0
                total = 0

                with torch.no_grad():
                    for examples, labels in test_loader:

                        # 改变数据类型
                        examples, labels = examples.to(device = self.device,dtype=torch.float), labels.to(
                            self.device)

                        outputs = self.model(examples)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy


    # 重写data loader，更改examples.to(device)的数据类型
    async def server_test(self, testset, sampler=None):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        # logging.info("plato->trainers basic.py server_test()")

        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        self.model.to(self.device)
        self.model.eval()

        custom_test = getattr(self, "test_model", None)

        if callable(custom_test):
            return self.test_model(config, testset)

        if sampler is None:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=config['batch_size'], shuffle=False)
        # Use a testing set following the same distribution as the training set
        else:
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['batch_size'],
                shuffle=False,
                sampler=sampler.get())

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:

                # 改变数据类型
                examples, labels = examples.to(device = self.device,dtype=torch.float), labels.to(
                    self.device)

                outputs = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Yield to other tasks in the server
                await asyncio.sleep(0)

        return correct / total



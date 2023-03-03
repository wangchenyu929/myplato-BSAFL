import logging
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

sys.path.append("..")
from plato.config import Config
from plato.clients import simple
from plato.utils import optimizers

@dataclass
class Report(simple.Report):
    """A client report containing the valuation, to be sent to the AFL federated learning server."""



class Client(simple.Client):

    

    # 改变返回值中sample数量的表示方法
    async def train(self):
        """The machine learning training workload on a client."""
        # logging.info("plato->clients simple.py train()")

        logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset, self.test_set_sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                self.client_id, 100 * accuracy))
        else:
            accuracy = 0

        data_loading_time = 0

        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True
        
        # 改变训练集数量的表示方法
        return Report(len(self.trainset),accuracy, training_time,
                      data_loading_time), weights

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import sys

import HARBox_dataloader
sys.path.append("..")
from plato.config import Config
from plato.servers import fedavg
from plato.utils import optimizers
from plato.processors import registry as processor_registry
from plato.utils import csv_processor


class Server(fedavg.Server):

    # 重新配置testset
    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        logging.info("[Server #%d] Configuring the server...", os.getpid())

        total_rounds = Config().trainer.rounds
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy:
            logging.info("Training: %s rounds or %s%% accuracy\n",
                         total_rounds, 100 * target_accuracy)
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.load_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer)

        if not Config().clients.do_test:
            # server端的测试集，用client0的全部训练数据
            self.testset = HARBox_dataloader.HARBox(train = True, client_id=0)

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)





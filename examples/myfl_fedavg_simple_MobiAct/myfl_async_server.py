import logging
import os
import numpy as np
import torch
import torch.nn as nn
import sys

import MobiAct_dataloader
sys.path.append("..")
from plato.config import Config
from plato.servers import fedavg
from plato.utils import optimizers
from plato.processors import registry as processor_registry
from plato.utils import csv_processor


class Server(fedavg.Server):

    def __init__(self, model=None, algorithm=None, trainer=None):
        # logging.info("########myfl_async_server.py init()########")
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.total_training_time = 0
    
    
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
            # server端的测试集，用client55的全部训练数据
            self.testset = MobiAct_dataloader.MobiAct(train = True, client_id=55)

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)

    # 重新定义round_time
    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        #logging.info("plato->servers fedavg.py wrap_up_processing_reports()")
        print("selected clients:",self.selected_clients)
        if hasattr(Config(), 'results'):
            self.total_training_time = self.total_training_time+max(self.selected_clients)*2
  
            new_row = []
            for item in self.recorded_items:
                item_value = {
                    'round':
                    self.current_round,
                    'accuracy':
                    self.accuracy * 100,
                    'training_time':
                    max(self.selected_clients)*2,
                    'round_time':
                    self.total_training_time
                }[item]
                new_row.append(item_value)

            result_csv_file = Config().result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)






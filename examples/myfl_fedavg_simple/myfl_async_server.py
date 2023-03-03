import logging
import os
import numpy as np
import torch
import torch.nn as nn
import sys

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






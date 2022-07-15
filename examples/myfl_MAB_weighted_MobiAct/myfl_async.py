
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['config_file'] = './myfl_async_MNIST_lenet5.yml'

import myfl_async_server
import myfl_async_client
import myfl_async_trainer
import sys
sys.path.append("..")
from plato.trainers import basic

def main():

    trainer = myfl_async_trainer.Trainer()
    server = myfl_async_server.Server(trainer=trainer)
    client = myfl_async_client.Client(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()

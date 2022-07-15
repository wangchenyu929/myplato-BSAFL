
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['config_file'] = './myfl_async_MNIST_lenet5.yml'

import sys
sys.path.append("..")
from plato.trainers import basic
from plato.servers import fedavg
from plato.clients import simple
import myfl_async_client
import myfl_async_server

def main():

    trainer = basic.Trainer()
    server = myfl_async_server.Server(trainer=trainer)
    client = myfl_async_client.Client(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Starting point for a Plato federated learning training session.
"""
import os
import sys
sys.path.append("../..")
from plato.servers import registry as server_registry

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['config_file'] = 'data_hetero_femnist_lenet5.yml'

def main():
    """Starting point for a Plato federated learning training session. """
    server = server_registry.get()
    server.run()


if __name__ == "__main__":
    main()

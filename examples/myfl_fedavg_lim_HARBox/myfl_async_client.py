"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

from ast import Not
import asyncio
import socketio
import logging
import sys
import pickle
import time
import os
import HARBox_dataloader
import ctypes
import random

from aiohttp import payload_type
import multiprocessing as mp
from dataclasses import dataclass
from random import choice
from plato.utils import s3

import sys
sys.path.append("..")
from plato.clients import simple,base
from plato.config import Config

@dataclass
class Report(simple.Report):
    """A client report."""
    client_id:int

class ClientEvents(base.ClientEvents):
    # 重写了on_payload_done函数，让payload_done函数
    async def on_payload_done(self, data):
        #logging.info("########myfl_async_client.py class.ClientEvent on_payload_done()########")
        if 's3_key' in data:
            await self.plato_client.payload_done(data,
                                                 s3_key=data['s3_key'])
        else:
            await self.plato_client.payload_done(data)


class Client(simple.Client):

    def __init__(self,model=None,datasource=None,algorithm=None,trainer=None):
        logging.info("########myfl_async_client.py class.Client init()########")
        super().__init__(model=model,datasource=datasource,algorithm=algorithm,trainer=trainer)


    # 如果computing_time > TRound，则直接不用算了，在periodic task到达时返回一个消息即可
    # 如果computing_time > TRound，则要训练
    async def payload_done(self, data, s3_key=None) -> None:
        """ Upon receiving all the new payload from the server. """
        logging.info("my payload_done()")
        computing_time = 10*random.randint(1,10)
        # computing_time = 0
        #如果computing_time注定大于TRound，则根本不用开始训练
        if computing_time > data["TRound"] :
            logging.info("[Client #%d]Training time will be %d,so I can`t complete the training" ,data["id"],computing_time)
            self.server_payload = None
            await self.sio.emit('client_payload_done', self.client_id)
        
        else:
            payload_size = 0

            if s3_key is None:
                if isinstance(self.server_payload, list):
                    for _data in self.server_payload:
                        payload_size += sys.getsizeof(pickle.dumps(_data))
                elif isinstance(self.server_payload, dict):
                    for key, value in self.server_payload.items():
                        payload_size += sys.getsizeof(pickle.dumps({key: value}))
                else:
                    payload_size = sys.getsizeof(pickle.dumps(self.server_payload))
            else:
                self.server_payload = self.s3_client.receive_from_s3(s3_key)
                payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

            assert data['id'] == self.client_id

            logging.info(
                "[Client #%d] Received %s MB of payload data from the server.",
                data['id'], round(payload_size / 1024**2, 2))

            self.server_payload = self.inbound_processor.process(
                self.server_payload)
            self.load_payload(self.server_payload)
            self.server_payload = None

            report, payload = await self.train()
            
            if Config().is_edge_server():
                logging.info(
                    "[Server #%d] Model aggregated on edge server (client #%d).",
                    os.getpid(), data[id])
            else:
                logging.info("[Client #%d] Model trained.", data['id'])

            # Sending the client report as metadata to the server (payload to follow)
            await self.sio.emit('client_report', {'report': pickle.dumps(report)})

            # Sending the client training payload to the server
            await self.send(payload,data)


    # 主要是为了使函数调用clientEvent的继承类
    async def start_client(self) -> None:
        """ Startup function for a client. """
        logging.info("plato->clients base.py start_client()")
        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            # Contact one of the edge servers
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                self.edge_server_id = int(
                    Config().clients.per_round) + (self.client_id - 1) % int(
                        Config().algorithm.total_silos) + 1
            else:
                self.edge_server_id = int(Config().clients.total_clients) + (
                    self.client_id - 1) % int(
                        Config().algorithm.total_silos) + 1
            logging.info("[Client #%d] Contacting Edge server #%d.",
                         self.client_id, self.edge_server_id)
        else:
            await asyncio.sleep(5)
            logging.info("[Client #%d] Contacting the central server.",
                         self.client_id)

        self.sio = socketio.AsyncClient(reconnection=True)
        #将此函数放在my_async_client文件中，调用的就是本文件中的clientEvents类了
        self.sio.register_namespace(
            ClientEvents(namespace='/', plato_client=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        if hasattr(Config().server, 'use_https'):
            uri = 'https://{}'.format(Config().server.address)
        else:
            uri = 'http://{}'.format(Config().server.address)

        if hasattr(Config().server, 'port'):
            # If we are not using a production server deployed in the cloud
            if hasattr(Config().algorithm,
                       'cross_silo') and not Config().is_edge_server():
                uri = '{}:{}'.format(
                    uri,
                    int(Config().server.port) + int(self.edge_server_id))
            else:
                uri = '{}:{}'.format(uri, Config().server.port)

        logging.info("[Client #%d] Connecting to the server at %s.",
                     self.client_id, uri)
        await self.sio.connect(uri)
        await self.sio.emit('client_alive', {'id': self.client_id})

        logging.info("[Client #%d] Waiting to be selected.", self.client_id)
        await self.sio.wait()


    async def send(self, payload, model_data) -> None:
        """Sending the client payload to the server using either S3 or socket.io."""
        logging.info("my send()")

        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        metadata = model_data

        if self.s3_client is not None:
            unique_key = uuid.uuid4().hex[:6].upper()
            s3_key = f'client_payload_{self.client_id}_{unique_key}'
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata['s3_key'] = s3_key
        else:
            if isinstance(payload, list):
                data_size: int = 0

                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data)
                    data_size += sys.getsizeof(_data)
            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data)
                data_size = sys.getsizeof(_data)

        await self.sio.emit('client_payload_done', metadata)

        logging.info("[Client #%d] Sent %s MB of payload data to the server.",
                     self.client_id, round(data_size / 1024**2, 2))


    ####### 以下是更改数据集所需的函数 #######
    
    # 将trainset和testset换成自定义的数据集
    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        logging.info("my load_data()")

        data_loading_start_time = time.perf_counter()
        logging.info("[Client #%d] Loading its data source...", self.client_id)

        # 自定义数据集
        self.trainset = HARBox_dataloader.HARBox(train = True, client_id=self.client_id)
        self.data_loaded = True

        logging.info("[Client #%d] Trainset loaded", self.client_id)


        if Config().clients.do_test:
            # Set the testset if local testing is needed
            # 自定义数据集
            self.testset = HARBox_dataloader.HARBox(train = False, client_id=self.client_id)
            logging.info("[Client #%d] Testset loaded", self.client_id)

        self.data_loading_time = time.perf_counter() - data_loading_start_time

    # 改变返回值中sample数量的表示方法
    async def train(self):
        """The machine learning training workload on a client."""
        logging.info("plato->clients simple.py train()")

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
                      data_loading_time,self.client_id), weights




"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

import asyncio
import socketio
import logging
import sys
import pickle
import time
import os
import random

import MobiAct_dataloader
import MobiAct_noniid
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
    model_round:int

class ClientEvents(base.ClientEvents):

    # server给client发送完了模型数据和当前轮数的信息
    async def on_payload_done(self, data):
        # 接收到的data数据如下
        # data = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}
        # logging.info("on_payload_done is on")
        """ All of the new payload sent from the server arrived. """
        if 's3_key' in data:
            await self.plato_client.payload_done(data['id'],
                                                 s3_key=data['s3_key'])
        else:
            # 参数由data['id']变为data
            # await self.plato_client.payload_done(data['id'])
            await self.plato_client.payload_done(data)
    

class Client(simple.Client):

    def __init__(self,model=None,datasource=None,algorithm=None,trainer=None):
        # logging.info("########myfl_async_client.py class.Client init()########")
        super().__init__(model=model,datasource=datasource,algorithm=algorithm,trainer=trainer)

        # 1代表当前空闲，0代表当前还在训练
        self.idle = 1
        # 预计训练几轮
        self.finish_round = 0
        # 理论上self.reality_finish中只会存一个模型
        # 用来临时存实际上训练完但理论上没有训练完的model
        # reality_finish 应当是一个字典
        # {'finish_round':self.finish_round, 'model_message': [data,report, payload]}
        self.reality_finish = {}

    # 使自己写的 ClientEvents 有效
    async def start_client(self) -> None:
        """ Startup function for a client. """
        # logging.info("plato->clients base.py start_client()")
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

        # logging.info("[Client #%d] Connecting to the server at %s.",
        #              self.client_id, uri)
        await self.sio.connect(uri)
        await self.sio.emit('client_alive', {'id': self.client_id})

        # logging.info("[Client #%d] Waiting to be selected.", self.client_id)
        await self.sio.wait()
    
    
    # 分空闲与不空闲两种情况
    async def payload_done(self, data, s3_key=None) -> None:
        """ Upon receiving all the new payload from the server. """
        # logging.info("my payload_done()")
        # 接收到的data数据如下
        # data = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}
        
        # 0代表忙碌，即client不空闲的情况
        if self.idle == 0:
            if data['current_round'] == self.finish_round:
                await self.sio.emit('client_report', {'report': pickle.dumps(self.reality_finish['model_message'][1])})
                await self.send(self.reality_finish['model_message'][2])
                # 发送的data数据就是从前接收到的data数据，其中包含model开始训练的轮数
                # data = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}
                await self.sio.emit('client_payload_done', self.reality_finish['model_message'][0])
                self.idle = 1
                self.reality_finish = {}
                # logging.info("client #%d send model version:%d",self.client_id,data['current_round'])
            else:

                await self.sio.emit('client_payload_done', self.client_id)
                logging.info("[Client #%d] Client is still training" ,self.client_id)
            # 根据当前轮数更新buffer
            self.server_payload = None
            # await self.update_send_buffer(data)
       
        # 1代表空闲，则让client训练，训练完后更新buffer，然后发送
        elif self.idle == 1:
            computing_time = 10*self.client_id
            # computing_time = 10*random.randint(1,10)
            self.finish_round = int(data['current_round']+(computing_time/data["TRound"]))
            if computing_time%data["TRound"] == 0:
                self.finish_round-=1
            # logging.info("[Client #%d] computing_time:%d,finish_round:%d",self.client_id,computing_time,self.finish_round)
            # 开始训练了，所以设置为忙碌
            self.idle = 0
            
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

            # assert data['client_id'] == self.client_id

            # logging.info(
            #     "[Client #%d] Received %s MB of payload data from the server.",
            #     self.client_id, round(payload_size / 1024**2, 2))

            self.server_payload = self.inbound_processor.process(
                self.server_payload)
            self.load_payload(self.server_payload)
            self.server_payload = None

            report, payload = await self.train(data['current_round'])

            # logging.info("[Client #%d] Model reality trained.", self.client_id)
            
            # 如果这一轮就能训练完，则直接发送
            if data['current_round'] == self.finish_round:
                await self.sio.emit('client_report', {'report': pickle.dumps(report)})
                await self.send(payload)
                # 发送的data数据就是接收到的data数据，其中包含model开始训练的轮数
                # data = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}
                await self.sio.emit('client_payload_done', data)
                self.idle = 1
                # logging.info("client #%d send model version:%d",self.client_id,data['current_round'])
            # 如果这一轮训练不完则先存到reality_finish中
            else:    
                await self.sio.emit('client_payload_done', self.client_id)
                self.reality_finish.update({'finish_round':self.finish_round, 'model_message': [data, report, payload]})

            
    # 在report中加入了client_id
    async def train(self,model_round):
        """The machine learning training workload on a client."""
        # logging.info("my train()")

        # logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        #是否需要测试本地模型准确率
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

        #是否需要发送数据加载时间
        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        return Report(len(self.trainset), accuracy, training_time,
                      data_loading_time,self.client_id,model_round), weights
    
    
    async def send(self, payload) -> None:
        """Sending the client payload to the server using either S3 or socket.io."""
        # logging.info("my send()")

        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

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

        # logging.info("[Client #%d] Sent %s MB of payload data to the server.",
        #              self.client_id, round(data_size / 1024**2, 2))

    ###### 更换数据集相关函数 ######
    # 将trainset和testset换成自定义的数据集
    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        # logging.info("my load_data()")

        data_loading_start_time = time.perf_counter()
        # logging.info("[Client #%d] Loading its data source...", self.client_id)

        # 自定义数据集
        self.trainset = MobiAct_noniid.MobiAct(train = True, client_id=self.client_id)
        self.data_loaded = True

        # logging.info("[Client #%d] Trainset loaded", self.client_id)


        if Config().clients.do_test:
            # Set the testset if local testing is needed
            # 自定义数据集
            self.testset = MobiAct_noniid.MobiAct(train = False, client_id=self.client_id)
            logging.info("[Client #%d] Testset loaded", self.client_id)

        self.data_loading_time = time.perf_counter() - data_loading_start_time






    





    
"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/abs/1812.07108
"""

import logging
import os
import MobiAct_dataloader
import asyncio
import pickle
import sys
import time

import socketio
import torch.nn.functional as F

from aiohttp import web
from collections import OrderedDict
from select import select

import sys
sys.path.append("..")
from plato.servers import fedavg,base
from plato.config import Config
from plato.utils import csv_processor
from plato.utils import s3
from plato.processors import registry as processor_registry

class ServerEvents(base.ServerEvents):

    async def on_client_response(self, sid, data):
        # logging.info("########myfl_async_server.py class.ServerEvents on_client_buffer_response()########")
        await self.plato_server.client_response(data)


    async def on_client_payload_done(self, sid, data):
        
        # 此时client接收到的消息是
        # metadata = {'id': client_id, "TRound" : self.TRound, "current_round": self.current_round}
        """ An existing client finished sending its payloads from local training. """

        # await self.plato_server.client_payload_done(sid, data['id'])
        await self.plato_server.client_payload_done(sid, data)

    

class Server(fedavg.Server):
    """ A federated learning server using the FedAtt algorithm. """    

    def __init__(self, model=None, algorithm=None, trainer=None):
        # logging.info("########myfl_async_server.py init()########")
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # 这是模拟的Tround
        self.TRound = 100
        # 用来记录模拟训练时间
        self.total_training_time = 0
        # 聚合的总client数
        self.total_aggregation_clients = 0
        # 已收到回应的client
        self.response_clients = []


    # 重新定义round_time
    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        # logging.info("plato->servers fedavg.py wrap_up_processing_reports()")
        if hasattr(Config(), 'results'):
            #self.total_training_time = self.total_training_time+(time.perf_counter() - self.round_start_time)
            self.total_training_time = self.total_training_time + self.TRound
            #local_time = time.strftime("%H:%M:%S", time.localtime()) 
            new_row = []
            for item in self.recorded_items:
                item_value = {
                    'round':
                    self.current_round,
                    'accuracy':
                    self.accuracy * 100,
                    'training_time':
                    max([
                        report.training_time for (report, __) in self.updates
                    ]),
                    'round_time':
                    self.total_training_time
                }[item]
                new_row.append(item_value)

            result_csv_file = Config().result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)

    # 给client发送payload done信息时要把 self.TRound 也发送过去
    async def send(self, sid, payload, client_id) -> None:
        """ Sending a new data payload to the client using either S3 or socket.io. """
        # logging.info("plato->servers base.py class.server send()")
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        # 在metadata中加入TRound
        metadata = {'id': client_id, "TRound" : self.TRound, "current_round": self.current_round}

        if self.s3_client is not None:
            s3_key = f'server_payload_{os.getpid()}_{self.current_round}'
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata['s3_key'] = s3_key
        else:
            data_size = 0

            if isinstance(payload, list):
                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data, sid, client_id)
                    data_size += sys.getsizeof(_data)

            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data, sid, client_id)
                data_size = sys.getsizeof(_data)

        await self.sio.emit('payload_done', metadata, room=sid)

        logging.info("[Server #%d] Sent %s MB of payload data to client #%d.",
                     os.getpid(), round(data_size / 1024**2, 2), client_id)

    # 让ServerClients用本文件中的class初始化
    def start(self, port=Config().server.port):
        """ Start running the socket.io server. """
        # logging.info("plato->servers base.py class.server start()")
        logging.info("Starting a server at address %s and port %s.",
                     Config().server.address, port)

        ping_interval = Config().server.ping_interval if hasattr(
            Config().server, 'ping_interval') else 3600
        ping_timeout = Config().server.ping_timeout if hasattr(
            Config().server, 'ping_timeout') else 360
        self.sio = socketio.AsyncServer(ping_interval=ping_interval,
                                        max_http_buffer_size=2**31,
                                        ping_timeout=ping_timeout)
        self.sio.register_namespace(
            ServerEvents(namespace='/', plato_server=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        app = web.Application()
        self.sio.attach(app)
        web.run_app(app,
                    host=Config().server.address,
                    port=port,
                    loop=asyncio.get_event_loop())


    # 把只接收client_id改为接受data，其中有model是用第几轮的全局model训练的信息   
    async def client_payload_done(self, sid, data, s3_key=None):
        # logging.info("my client_payload_done()")
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
        if isinstance(data,int):
            self.response_clients.append(data)
        else:
            # 当发送内容是字典时代表有模型发过来，data内容如下：
            # data = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}
            self.response_clients.append(data['id'])

            if s3_key is None:
                assert self.client_payload[sid] is not None

                payload_size = 0
                if isinstance(self.client_payload[sid], list):
                    for _data in self.client_payload[sid]:
                        payload_size += sys.getsizeof(pickle.dumps(_data))
                else:
                    payload_size = sys.getsizeof(
                        pickle.dumps(self.client_payload[sid]))
            else:
                self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
                payload_size = sys.getsizeof(pickle.dumps(
                    self.client_payload[sid]))

            logging.info(
                "[Server #%d] Received %s MB of payload data from client #%d.",
                os.getpid(), round(payload_size / 1024**2, 2), data['id'])

            self.client_payload[sid] = self.inbound_processor.process(
                self.client_payload[sid])
            self.updates.append((self.reports[sid], self.client_payload[sid]))

            self.reporting_clients.append(data['id'])
            del self.training_clients[data['id']]
                    # 如果收到了所有的client的消息，那么就从buffer中挑模型放入updates聚合
        logging.info(
                "[Server #%d] self.response_clients:",
                os.getpid())
        logging.info(self.response_clients)
        if len(self.response_clients) == self.clients_per_round:
            self.total_aggregation_clients += len(self.updates)
            
            logging.info(
                "[Server #%d] All %d client reports received. Processing.Total aggregation clients:%d",
                os.getpid(), len(self.response_clients),self.total_aggregation_clients)
            self.response_clients = []
            if len(self.updates) != 0:
                await self.process_reports()
            else:
                await self.write_results_in_file()
            await self.wrap_up()
            await self.select_clients()


    # 当没有model上传时，向csv文件中写的内容
    async def write_results_in_file(self):
       
        self.total_training_time = self.total_training_time + self.TRound
        
        result_csv_file = Config().result_dir + 'result.csv'
        new_row = [self.current_round,self.accuracy * 100,self.total_training_time]
        csv_processor.write_csv(result_csv_file, new_row)


    ####### 以下是更改数据集所需的函数 #######
    
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






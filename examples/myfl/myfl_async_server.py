"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/abs/1812.07108
"""

import logging
import os
import time
import asyncio
import pickle
import sys
import random
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

class ServerEvents(base.ServerEvents):

    # async def on_client_training_done(self, sid, data):
    #     logging.info("########myfl_async_server.py class.ServerEvents on_client_training_done()########")
    #     await self.plato_server.client_training_done(sid,data)

    async def on_client_response(self, sid, data):
        logging.info("########myfl_async_server.py class.ServerEvents on_client_buffer_response()########")
        await self.plato_server.client_response(sid,data)

    # 接收时间片达到后从client传递过来的buffer和idel消息
    async def on_client_idle_buffer_message_done(self, sid, data):
        # logging.info("########myfl_async_server.py class.ServerEvents on_client_idle_buffer_message_done########")
        await self.plato_server.client_idle_buffer_message_done(sid,data)
    

class Server(fedavg.Server):
    """ A federated learning server using the FedAtt algorithm. """    

    def __init__(self, model=None, algorithm=None, trainer=None):
        logging.info("########myfl_async_server.py init()########")
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.selected_buffer = None
        self.buffer_per_round = 10
        # 在挑选buffer聚合时
        self.client_buffer_response=[]
        # 当前轮空闲的clients
        self.idle_clients=[]
        # 用来记录每个client的buffer内容
        self.client_buffer_len=[]

    
    def choose_clients(self, selectable_clients, clients_count):
        """ 选择被分发模型的client，选择策略为给所有空闲的client分发模型使其训练 """
        logging.info("########myfl_async_server.py choose_clients()########")
        #clients pool里是当前空闲的clients
        logging.info("########selectable clients########")
        logging.info(selectable_clients)
        return selectable_clients


    async def periodic_task(self):
        logging.info("########myfl_async_server.py periodic_task()########")
        #如果是从buffer中取模型那就不用检查模型版本号，因为client会在buffer中就检查版本号
        #默认是异步FL
        #if self.selected_clients != None and len(self.selected_clients) >0 :
        if len(self.clients_pool)>0 :
            
            #logging.info("[Server #%d] get model from clients` buffer",os.getpid())
            #从client的buffer中get模型
            await self.get_buffer_models()

            #这里需要等client传回消息再进行下一步
            while len(self.client_buffer_response)!=len(self.clients_pool):
                await asyncio.sleep(0.5)
                logging.info("periodic task waiting for all clients response")
                pass
            
            logging.info("########client_buffer_response[]########")
            logging.info(self.client_buffer_response)
            
            # client_buffer_response完成使命，将其归零
            self.client_buffer_response.clear()

            logging.info("########self.updates number########")
            logging.info(len(self.updates))
            if len(self.updates) > 0:
                for i,update in enumerate(self.updates):
                    logging.info("########self.updates id########")
                    report,__ = update
                    logging.info(report.client_id)
                
                #聚合收到的模型
                await self.process_reports()
                #判断终止条件
                await self.wrap_up()
                #下一轮选择clients
                await self.select_clients()
            else:
                logging.info("[Server #%d] No sufficient number of client reports have been received. ""Nothing to process.",os.getpid())
                # 这一轮没有可以聚合的model，那么就写上一轮的accuracy
                await self.write_results_in_file()
                self.current_round += 1
                logging.info("\n[Server #%d] Starting round %s/%s.", os.getpid(),
                        self.current_round,
                        Config().trainer.rounds)
        
        else:
            logging.info("[Server #%d] Training has not started.",os.getpid())
    
    
    async def send(self, sid, payload, client_id) -> None:
        """ server给client传递信息，增加传递本轮轮数的功能 """
        """ Sending a new data payload to the client using either S3 or socket.io. """
        # First apply outbound processors, if any
        logging.info("########myfl_async_server.py send()")
        payload = self.outbound_processor.process(payload)
        
        #在元数据中增加client刚接收到模型的轮数
        metadata = {'id': client_id, "round": self.current_round}
        logging.info("######## metadata: client_id:%d;client_round:%d",client_id,self.current_round)


        if self.s3_client is not None:
            s3_key = f'server_payload_{os.getpid()}_{self.current_round}'
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata['s3_key'] = s3_key
        else:
            data_size = 0
            #如果payload是list类型，则逐个发送
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


    # 当有被选中的client回复消息时，将client id加入self.client_buffer_response中
    async def client_response(self,sid,data):
        self.client_buffer_response.append(data)
        logging.info("########add client into client_buffer_response[]########")
        logging.info(self.client_buffer_response)


    async def client_idle_buffer_message_done(self, sid, data):
        #data = {"client_id":self.client_id,"is_idle":is_idle,"buffer_size":len(self.buffer)}
        if data["is_idle"] == 1:
            self.idle_clients.append(data["client_id"])
        # logging.info("client_idle_buffer_message_done idle_clients:")
        # logging.info(self.idle_clients)
        self.client_buffer_len.append({"client_id":data["client_id"],"buffer_size":data["buffer_size"]})
        # logging.info("client_idle_buffer_message_done client_buffer_len:")
        # logging.info(self.client_buffer_len)


    # server需要给所有client发消息，
    # 等接收到所有client返回的buffer信息、空闲信息后再更改idle_clients
    # 并且给被选择的client发消息让其返回buffer信息
    async def get_buffer_models(self):
        #logging.info("########myfl_async_server.py get_buffer_models()########")

        #给所有的client发送消息告诉它们时间片到了
        for client_id in self.clients_pool:
            sid = self.clients[client_id]['sid']
            server_response = {'id': client_id,'current_round':self.current_round}
            await self.sio.emit('client_message_to_arrive',{'response': server_response},room=sid)
        # 要等所有client都回复消息再进行下一步
        while len(self.client_buffer_len)!=len(self.clients_pool):
            
            logging.info("get buffer waiting for all clients response")
            await asyncio.sleep(0.5)
            pass
        for buffer_msg in self.client_buffer_len:
            logging.info("buffer_msg:%d",buffer_msg['client_id'])
        #可以对比两种选择策略，一种取最多的buffer中的model，一种随机取
        #随机取
        client_count=0 #用来记录当前已经取了多少个model了
        for i,selected_client in enumerate(self.client_buffer_len):
            # 如果buffer中有内容
            if selected_client["buffer_size"]!=0:
                server_response = {'id': selected_client["client_id"]}
                sid = self.clients[selected_client["client_id"]]['sid']
                # room参数是指向特定的client发送消息
                await self.sio.emit('buffer_select_to_arrive',{'response': server_response},room=sid)
                logging.info("[Server #%d] Sending the select buffer message to client #%d.",os.getpid(), selected_client["client_id"])
                client_count+=1
            if client_count==self.buffer_per_round:
                break

        #如果buffer中没有足够的model，则用0补齐
        for i in range(0,len(self.clients_pool)-client_count):
            self.client_buffer_response.append(0)
            
        #已经给所有client发送完，更新self.client_buffer_len
        self.client_buffer_len=[]


    # 选择buffer暂定为随机选择
    # 暂时没用了，已经在get_buffer_models中完成选择
    def choose_buffer(self, clients_pool, buffer_count):
        """ Choose a subset of the clients to participate in each round. """
        #因为第一轮的client pool
        logging.info("########myfl_async_server.py choose_buffer########")
        #assert buffer_count <= len(clients_pool)
        # Select clients randomly
        # 只有当client_pool中的client数不为0时才选择
        if len(clients_pool) > buffer_count:
            selected_buffer = random.sample(clients_pool, buffer_count)
            logging.info("########clients_pool:")
            logging.info(clients_pool)
            logging.info("########len(clients_pool) > buffer_count:")
            logging.info(selected_buffer)
            return selected_buffer
        #这个else是指client_pool中无
        else:
            return []


    #把删除training_client的语句去除
    async def client_payload_done(self, sid, client_id, s3_key=None):
        logging.info("########myfl_async_server.py class.server client_payload_done()")
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
        self.client_buffer_response.append(client_id)
        # logging.info("########add client into client_buffer_response[]########")
        # logging.info(self.client_buffer_response)

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
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        self.client_payload[sid] = self.inbound_processor.process(
            self.client_payload[sid])
        self.updates.append((self.reports[sid], self.client_payload[sid]))

        self.reporting_clients.append(client_id)
        #del self.training_clients[client_id]

        # If all updates have been received from selected clients, the aggregation process
        # proceeds regardless of synchronous or asynchronous modes. This guarantees that
        # if asynchronous mode uses an excessively long aggregation interval, it will not
        # unnecessarily delay the aggregation process.
        if len(self.updates) >= self.clients_per_round:
            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.updates))
            await self.process_reports()
            await self.wrap_up()
            await self.select_clients()
  
    
    #这里是选择每一轮server给分发模型的client，需要选择那些空闲的client
    async def select_clients(self):
        logging.info("plato->servers base.py class.server select_clients()")
        """ Select a subset of the clients and send messages to them to start training. """
        self.updates = []
        self.current_round += 1

        logging.info("\n[Server #%d] Starting round %s/%s.", os.getpid(),
                     self.current_round,
                     Config().trainer.rounds)

        if hasattr(Config().clients, 'simulation') and Config(
        ).clients.simulation and not Config().is_central_server:
            # In the client simulation mode, the client pool for client selection contains
            # all the virtual clients to be simulated
            self.clients_pool = list(range(1, 1 + self.total_clients))

        else:
            # If no clients are simulated, the client pool for client selection consists of
            # the current set of clients that have contacted the server
            self.clients_pool = list(self.clients)

        # In asychronous FL, avoid selecting new clients to replace those that are still
        # training at this time
        #如果配置文件中设置训练方式为异步，则每轮选择空闲的client来参加训练
        if hasattr(Config().server, 'synchronous') and not Config().server.synchronous:
            #如果是第一轮，则selected_clients就是全部的client
            if self.current_round == 1:
                self.selected_clients = self.clients_pool
            
            #如果不是第一轮，则根据clients传回的消息看空闲的clients
            else:
                self.selected_clients = self.idle_clients
            #选择完毕，更新self.idle_clients
            self.idle_clients=[]
        
        #训练方式为同步，则每轮从client_pool中选择指定数量的client
        else:
            self.selected_clients = self.choose_clients(
                self.clients_pool, self.clients_per_round)
        
        logging.info("selected_clients:")
        logging.info(self.selected_clients)

        if len(self.selected_clients) > 0:
            for i, selected_client_id in enumerate(self.selected_clients):
                if hasattr(Config().clients, 'simulation') and Config(
                ).clients.simulation and not Config().is_central_server:
                    if hasattr(Config().server, 'synchronous') and not Config(
                    ).server.synchronous and self.reporting_clients is not None:
                        client_id = self.reporting_clients[i]
                    else:
                        client_id = i + 1
                else:
                    client_id = selected_client_id

                sid = self.clients[client_id]['sid']

                logging.info("[Server #%d] Selecting client #%d for training.",
                             os.getpid(), selected_client_id)

                server_response = {'id': selected_client_id}
                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                # Sending the server payload to the client
                logging.info(
                    "[Server #%d] Sending the current model to client #%d.",
                    os.getpid(), selected_client_id)
                await self.send(sid, payload, selected_client_id)

                self.training_clients[client_id] = {
                    'id': selected_client_id,
                    'round': self.current_round
                }

            self.reporting_clients = []


    #让ServerClients用本文件中的class初始化
    def start(self, port=Config().server.port):
        """ Start running the socket.io server. """
        logging.info("plato->servers base.py class.server start()")
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

    # 当没有model上传时，向csv文件中写的内容
    async def write_results_in_file(self):
        #self.total_training_time = self.total_training_time+(time.perf_counter() - self.round_start_time)
        local_time = time.strftime("%H:%M:%S", time.localtime()) 
        result_csv_file = Config().result_dir + 'result.csv'
        new_row = [self.current_round,self.accuracy * 100,local_time]
        csv_processor.write_csv(result_csv_file, new_row)
    
    
    # 重新定义round_time
    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        logging.info("plato->servers fedavg.py wrap_up_processing_reports()")
        if hasattr(Config(), 'results'):
            #self.total_training_time = self.total_training_time+(time.perf_counter() - self.round_start_time)
            local_time = time.strftime("%H:%M:%S", time.localtime()) 
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
                    local_time
                }[item]
                new_row.append(item_value)

            result_csv_file = Config().result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)
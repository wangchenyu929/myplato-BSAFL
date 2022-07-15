import logging
import os
import asyncio
import pickle
import sys
import numpy as np
import socketio
import torch.nn.functional as F
import torch
import copy
import math

from aiohttp import web
import MobiAct_dataloader
import sys
sys.path.append("..")
from plato.servers import fedavg,base
from plato.config import Config
from plato.utils import csv_processor
from plato.processors import registry as processor_registry

class ServerEvents(base.ServerEvents):

    async def on_client_payload_done(self, sid, data):
        # logging.info("my on_client_payload_done()")
        # 此时client传回的消息是
        # metadata = {'id': client_id, "TRound" : self.TRound, "current_round": self.current_round}
        """ An existing client finished sending its payloads from local training. """
        # await self.plato_server.client_payload_done(sid, data['id'])
        await self.plato_server.client_payload_done(sid, data)
    

class Server(fedavg.Server):    

    def __init__(self, model=None, algorithm=None, trainer=None):
        # logging.info("########myfl_async_server.py init()########")
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        # server理论上的时间片值
        self.TRound = 200
        # 用来记录当前轮所有client的buffer状态
        # 因为将buffer放在每个client端会因为通信问题准确率不增长，所以暂时放在server端
        self.client_buffer = []
        # 因为下标是从0开始的，但是client id是从1开始的，所以把第0个位置空出来
        for i in range(self.total_clients+1):
            self.client_buffer.append([])
        # 每轮要聚合的client数
        self.buffer_per_round = 10
        # 没有训练完，但已收到回应的client
        self.response_uncomplete = []
        # 已训练完，并且已收到回应的client
        self.response_complete = []

        # 记录总的聚合时间
        self.total_training_time = 0
        # stalness
        self.staleness = 0
        if hasattr(Config().server, 'staleness_bound'):
                self.staleness = Config().server.staleness_bound

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
    
    # 每一轮都给所有client发送消息，顺便让client返回buffer情况
    async def select_clients(self):
        # logging.info("my select_clients()")
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

        # 给所有client都发送当前模型
        self.selected_clients = self.clients_pool

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

                # logging.info("[Server #%d] Selecting client #%d for training.",
                #              os.getpid(), selected_client_id)

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
                # logging.info(
                #     "[Server #%d] Sending the current model to client #%d.",
                #     os.getpid(), selected_client_id)
                await self.send(sid, payload, selected_client_id)

                self.training_clients[client_id] = {
                    'id': selected_client_id,
                    'round': self.current_round
                }

            self.reporting_clients = []

    # 不仅要发送payload，还要发送当前轮数和每轮的时间片
    async def send(self, sid, payload, client_id) -> None:
        """ Sending a new data payload to the client using either S3 or socket.io. """
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        # 在metasdata中加入了current_round和TRound
        metadata = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}

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

        # logging.info("[Server #%d] Sent %s MB of payload data to client #%d.",
        #              os.getpid(), round(data_size / 1024**2, 2), client_id)

    # data有可能是一个字典（当有模型发送过来）也有可能是一个int型（当没有训练完的模型）
    async def client_payload_done(self, sid, data, s3_key=None):
        # logging.info("my client_payload_done()")
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
        if isinstance(data,int):
            self.response_uncomplete.append(data)
        else:
            # 当发送内容是字典时代表有模型发过来，data内容如下：
            # data = {'id': client_id,'current_round':self.current_round, 'TRound':self.TRound}
            self.response_complete.append(data['id'])

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

            # 给对应的client添加buffer
            # 因为list下标是从0开始的
            self.client_buffer[data['id']].append([self.reports[sid],data,self.client_payload[sid]])

            # self.updates.append((self.reports[sid], self.client_payload[sid]))

            # self.reporting_clients.append(data['id'])
            del self.training_clients[data['id']]

        # 如果收到了所有的client的消息，那么就从buffer中挑模型放入updates聚合
        if len(self.response_uncomplete)+ len(self.response_complete)== len(self.clients_pool):
            print("All client reports received. Processing",self.response_complete)
            
            # 要从buffer中选模型
            await self.select_model()
            self.response_uncomplete = []
            self.response_complete = []
      
            if len(self.updates) != 0:
                await self.process_reports()
            await self.wrap_up()
            await self.select_clients()

    # 从buffer中挑模型
    async def select_model(self):
        buffer_client_id = -1
        buffer_select = []
        training_staleness = []
        waiting_staleness = []
        # 需要处理staleness
        for client_buffer in self.client_buffer:
            
            buffer_client_id+=1

            # 先处理staleness
            for buffer_message in client_buffer:
                # 此模型超过staleness
                # logging.info("before stalness:client %d`s buffer length:%d",buffer_message[1]['id'],len(self.client_buffer[buffer_message[1]['id']]))
                #随机生成每个client的staleness bound
                staleness_bound = self.staleness+int((buffer_client_id*10)/self.TRound)
                if self.current_round-buffer_message[1]['current_round']>staleness_bound:
                # if self.current_round-buffer_message[1]['current_round']>self.staleness:
                    
                    # logging.info("client %d has a stale model,which version is :%d,stalness bound is %d",buffer_message[1]['id'],buffer_message[1]['current_round'],staleness_bound)
                    # logging.info("client %d has a stale model,which version is :%d",buffer_message[1]['id'],buffer_message[1]['current_round'])
                    client_buffer.remove(buffer_message)
                    # logging.info("after stalness:client %d`s buffer length:%d",buffer_message[1]['id'],len(self.client_buffer[buffer_message[1]['id']]))

            # 更新waiting staleness
            if len(client_buffer) != 0:
                buffer_select.append(buffer_client_id)
                # training_staleness是training time/TRound取上届再减1
                training_staleness.append(int(math.ceil(buffer_client_id*10/self.TRound))-1)
                # 都是取最新的模型
                # waiting staleness是总staleness - waiting staleness
                w_s = self.current_round-client_buffer[-1][1]['current_round']-training_staleness[-1]
                waiting_staleness.append(w_s)
        # 按照概率选择client
        print("waiting_staleness:",waiting_staleness)
        print("training_staleness:",training_staleness)
        print("buffer_select:",buffer_select)
        probs = [
            (training_staleness[i]+1)/(waiting_staleness[i]+1)
            for i in range(len(buffer_select))
            ]
        # 归一化
        probs = np.array(probs)
        probs /= sum(probs)
        print("probility:",probs)

        if len(buffer_select) <= self.buffer_per_round:
            buffer_selected_client = buffer_select
        else:
            buffer_selected_client = np.random.choice(buffer_select,
                                   self.buffer_per_round,
                                   p=probs,
                                   replace=False).tolist()
        print("buffer_selected_client:",buffer_selected_client)

        # 从已选择的client的buffer中取模型
        for client_id in buffer_selected_client:
            # 如果某个client的buffer不为0，则取出最新模型进行聚合
            # logging.info("client %d`s buffer length:%d",client_id,len(self.client_buffer[client_id]))
            if len(self.client_buffer[client_id]) != 0:
                update_model =  self.client_buffer[client_id].pop()
                self.updates.append((update_model[0],update_model[2]))
                self.reporting_clients.append(client_id)
                # logging.info("client %d`s buffer length:%d",client_id,len(self.client_buffer[client_id]))
                # logging.info("client %d`s model version:%d",client_id,update_model[1]['current_round'])

    # 当没有model上传时，向csv文件中写的内容
    async def write_results_in_file(self):
        self.total_training_time = self.total_training_time + self.TRound
        #local_time = time.strftime("%H:%M:%S", time.localtime()) 
        result_csv_file = Config().result_dir + 'result.csv'
        new_row = [self.current_round,self.accuracy * 100,self.total_training_time]
        csv_processor.write_csv(result_csv_file, new_row)

    # 重新定义round_time
    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        #logging.info("plato->servers fedavg.py wrap_up_processing_reports()")
        if hasattr(Config(), 'results'):
            self.total_training_time = self.total_training_time+self.TRound
  
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

    
    #Baochun Li method

    async def cosine_similarity(self, update, staleness):
        """ Compute the cosine similarity of the received updates and the difference
            between the current and a previous model according to client staleness. """
        # Loading the global model from a previous round according to staleness
        filename = f"model_{self.current_round - 2}.pth"
        model_dir = Config().params['model_dir']
        model_path = f'{model_dir}/{filename}'

        similarity = 1.0

        if staleness > 1 and os.path.exists(model_path):
            previous_model = copy.deepcopy(self.trainer.model)
            previous_model.load_state_dict(torch.load(model_path))

            previous = torch.zeros(0)
            for __, weight in previous_model.cpu().state_dict().items():
                previous = torch.cat((previous, weight.view(-1)))

            current = torch.zeros(0)
            for __, weight in self.trainer.model.cpu().state_dict().items():
                current = torch.cat((current, weight.view(-1)))

            deltas = torch.zeros(0)
            for __, delta in update.items():
                deltas = torch.cat((deltas, delta.view(-1)))

            similarity = F.cosine_similarity(current - previous, deltas, dim=0)

        return similarity

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report,__) in updates])

        # Constructing the aggregation weights to be used
        aggregation_weights = []

        for i, update in enumerate(weights_received):
            report, __ = updates[i]
            num_samples = report.num_samples
            staleness = self.current_round-report.model_round
            similarity = await self.cosine_similarity(update, staleness)
            staleness_factor = Server.staleness_function(staleness)

            similarity_weight = Config().server.similarity_weight if hasattr(
                Config().server, 'similarity_weight') else 1
            staleness_weight = Config().server.staleness_weight if hasattr(
                Config().server, 'staleness_weight') else 1

            # logging.info('[Client %d] similarity: %s', report.client_id, (similarity + 1) / 2)
            # logging.info('[Client %d] staleness: %s, staleness factor: %s', report.client_id, staleness, staleness_factor)
            raw_weight = num_samples / self.total_samples * (
                (similarity + 1) / 2 * similarity_weight +
                staleness_factor * staleness_weight)
            logging.info('[Client %d] raw weight = %s', report.client_id, raw_weight)

            aggregation_weights.append(raw_weight)

        # Normalize so that the sum of aggregation weights equals 1
        aggregation_weights = [
            i / sum(aggregation_weights) for i in aggregation_weights
        ]

        logging.info('All client normalized weights: %s',
                     aggregation_weights)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __= updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * aggregation_weights[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update


    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

        # Save the current model for later retrieval when cosine similarity needs to be computed
        filename = f"model_{self.current_round}.pth"
        self.trainer.save_model(filename)


    @staticmethod
    def staleness_function(staleness):
        """ The staleness_function. """
        staleness_bound = 10

        if hasattr(Config().server, "staleness_bound"):
            staleness_bound = Config().server.staleness_bound

        staleness_factor = staleness_bound / (staleness + staleness_bound)

        return staleness_factor

    ###### 更换数据集相关函数 ######
    
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
            self.testset = MobiAct_dataloader.MobiAct(train = True, client_id=55)

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)

    

    
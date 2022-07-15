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
import threading
import random


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
    #重写了on_payload_done函数，让payload_done函数
    async def on_payload_done(self, data):
        #logging.info("########myfl_async_client.py class.ClientEvent on_payload_done()########")
        if 's3_key' in data:
            await self.plato_client.payload_done(data,
                                                 s3_key=data['s3_key'])
        else:
            await self.plato_client.payload_done(data)

    async def on_buffer_select_to_arrive(self, data):
        #logging.info("########myfl_async_client.py class.ClientEvent buffer_select_to_arrive()########")
        await self.plato_client.buffer_select_to_arrive(data)

    #当时间片到达后server给每个client都发送消息，要求client返回当前的空闲状态以及buffer中model个数
    async def on_client_message_to_arrive(self, data):
        # logging.info("########myfl_async_client.py class.ClientEvent on_client_message_to_arrive()########")
        await self.plato_client.client_message_to_arrive(data)


class Client(simple.Client):

    def __init__(self,model=None,datasource=None,algorithm=None,trainer=None):
        logging.info("########myfl_async_client.py class.Client init()########")
        super().__init__(model=model,datasource=datasource,algorithm=algorithm,trainer=trainer)
        #self的buffer,是list结构，里面应当存放[[metadate,report,payload],[metadate,report,payload],…]
        #data的结构：["id":client_id;"round":current_round]
        self.buffer=[]
        self.training_thread = None

    # 这是client接收完server发来的模型信息立刻开始训练并且将模型发送到server的消息，
    # 将client_id参数改为data参数，data中包括“id”和“round”两个字典内容
    async def payload_done(self, data, s3_key=None) -> None:
        """ Upon receiving all the new payload from the server. """
        # logging.info("########myfl_async_client.py class.Client payload_done()########")
        # logging.info("type of training_thread")
        # logging.info(type(self.training_thread))
        if self.training_thread is None or not self.training_thread.is_alive():
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
        
            # logging.info("########data[id]:########")
            # logging.info(data["id"])
            logging.info("########self.client_id:########")
            logging.info(self.client_id)
            assert data["id"] == self.client_id

            logging.info(
                "[Client #%d] Received %s MB of payload data from the server.",
                data["id"], round(payload_size / 1024**2, 2))

            self.server_payload = self.inbound_processor.process(
                self.server_payload)
            
            self.load_payload(self.server_payload)
            self.server_payload = None
            try:

                #proc_train = mp.Process(target=self.train,args=(data,))
                #proc_train.start()

                #loop = asyncio.get_event_loop()
                self.training_thread = threading.Thread(target = self.train,args = (data,))
                self.training_thread.start()
            except:
                logging.info("can not start thread")
            logging.info("[Client #%d] Model is training.", data["id"])
       
        else:
            # 清除一下server发来的模型
            self.server_payload = None
            logging.info("[Client #%d] is still training.",data["id"])

    # 收到从server发来的消息，client应当从自身的buffer中取模型并发送给server
    async def buffer_select_to_arrive(self, data):
        # 从buffer中随机选取一个模型并从buffer中删除
        # logging.info("########[Client #%d] myfl_async_client.py class.Client buffer_select_to_arrive()",self.client_id)
        #buffer_model content:[data,report,payload]
        logging.info("########[Client #%d] self.buffer.length",self.client_id)
        logging.info(len(self.buffer))
        if len(self.buffer) > 0 :

            buffer_model = self.buffer[-1]
            del(self.buffer[-1])
            #self.buffer.remove(buffer_model)
            #将buffer中的report先发送给server
            await self.sio.emit('client_report', {'report': pickle.dumps(buffer_model[1])})
            # 发送payload
            # 如果payload是list类型，则逐个发送，否则直接发送
            if isinstance(buffer_model[2], list):
                    data_size: int = 0

                    for data in buffer_model[2]:
                        _data = pickle.dumps(data)
                        await self.send_in_chunks(_data)
                        data_size += sys.getsizeof(_data)
            else:
                    _data = pickle.dumps(buffer_model[2])
                    await self.send_in_chunks(_data)
                    data_size = sys.getsizeof(_data)
            await self.sio.emit('client_payload_done', buffer_model[0])

            logging.info("[Client #%d] Sent %s MB of buffer payload data to the server.",
                        self.client_id, round(data_size / 1024**2, 2))
        else:
            # 就算buffer没有内容也要给server一个答复
            await self.sio.emit('client_response', self.client_id)
            logging.info("[Client #%d] buffer no sufficient model",self.client_id)

    # 向server发送自身的空闲状态以及buffer中model个数    
    async def client_message_to_arrive(self, data):
        # 如果线程是None，则说明第一次训练还没有开始，如果线程处于不活跃状态，则说明已经训练完毕
        if self.training_thread is None or not self.training_thread.is_alive():
            is_idle = 1
        else:
            is_idle = 0
        send_data = {"client_id":self.client_id,"is_idle":is_idle,"buffer_size":len(self.buffer)}
        #logging.info("[Client #%d] client_message_to_arrive idle and buffer message:", self.client_id)
        #logging.info(send_data)
        await self.sio.emit('client_idle_buffer_message_done', send_data)



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
        
        
    # 是为了将client_id加到report中
    # 训练完成后马上给Server发送消息
    def train(self,data):
        """The machine learning training workload on a client."""
        sleep_time = random.randint(1,10)*4
        logging.info("[Client #%d]sleep time:%d",self.client_id,sleep_time)
        time.sleep(sleep_time)
        
        logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError:
            self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        #是否需要测试本地模型准确率
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset, self.test_set_sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                self.sio.disconnect()

            logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                self.client_id, 100 * accuracy))
        else:
            accuracy = 0

        data_loading_time = 0

        #是否需要发送数据加载时间
        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True
       
       
        logging.info("[Client #%d] Model trained.", data["id"])
        #这里需要加一个检验buffer中模型是否过期的函数
        logging.info("[Client #%d] put model into buffer.", data["id"])
        report = Report(self.sampler.trainset_size(), accuracy, training_time,
                      data_loading_time,self.client_id)
        payload = weights
        self.buffer.append([data,report,payload])
        logging.info("########client #%d `s buffer in round %d:",data["id"],data["round"])
        logging.info("########buffer length: %d",len(self.buffer))

        # 相当于给server发消息说自己训练完成
        # global_value.add_training_done_clients(data["id"])

        # logging.info("########:training_done_clients")
        # logging.info(global_value.get_training_done_clients())
        # train_done_message = {'client_id': data["id"], 'buffer_length': len(self.buffer)}
        # #loop = asyncio.get_running_loop()
        # asyncio.set_event_loop(loop)
        # loop.run_until_complete(self.send_train_done_message(train_done_message))
        
        #向server发送信息表明自己训练完成了，信息包括{'client_id': id ,'buffer_count': count}
        #await self.sio.emit('client_training_done', {'client_id': data["id"], 'buffer_length': len(self.buffer)})

        return report, payload



    
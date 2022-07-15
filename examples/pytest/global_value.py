
def _init():
    # 全局变量 已经训练完的client
    global training_done_clients 
    training_done_clients = []

def add_training_done_clients(value):
    # 在training_done_clients中逐个加入变量
    training_done_clients.append(value)

def del_training_done_clients(value):
    # 在training_done_clients中删除变量,需要传入的变量也需要是list类型
    for i in value:
        training_done_clients.remove(i)

def get_training_done_clients():
    return training_done_clients
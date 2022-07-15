import logging
import global_value
import logging

def main():
    global_value._init()
    global_value.add_training_done_clients([1,2,3])
    print("global1:")
    print(global_value.get_training_done_clients())



if __name__ == '__main__':
    main()


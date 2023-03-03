import csv
from tkinter import CENTER
from turtle import color
from typing import Dict, List, Any
import matplotlib.pyplot as plt


def read_csv_to_dict(result_csv_file_list):
    """Read a CSV file and write the values that need to be plotted
    into a dictionary."""
    result_dict1: Dict[str, List] = {}
    result_dict2: Dict[str, List] = {}
    result_dict3: Dict[str, List] = {}
    result_dict4: Dict[str, List] = {}

    # plot_pairs = Config().results.plot
    # plot_pairs = [x.strip() for x in plot_pairs.split(',')]
    plot_pairs = ['round&accuracy','round_time&accuracy']


    for pairs in plot_pairs:
        pair = [x.strip() for x in pairs.split('&')]
        for item in pair:
            if item not in result_dict1 and item not in result_dict2:
                result_dict1[item] = []
                result_dict2[item] = []
                result_dict3[item] = []
                result_dict4[item] = []

    with open(result_csv_file_list[0], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in result_dict1:
                if item in ('round', 'global_round'):
                    result_dict1[item].append(int(row[item]))
                else:
                    result_dict1[item].append(float(row[item]))
    with open(result_csv_file_list[1], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in result_dict2:
                if item in ('round', 'global_round'):
                    result_dict2[item].append(int(row[item]))
                else:
                    result_dict2[item].append(float(row[item]))
   
    with open(result_csv_file_list[2], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in result_dict3:
                if item in ('round', 'global_round'):
                    result_dict3[item].append(int(row[item]))
                else:
                    result_dict3[item].append(float(row[item]))

    with open(result_csv_file_list[3], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in result_dict4:
                if item in ('round', 'global_round'):
                    result_dict4[item].append(int(row[item]))
                else:
                    result_dict4[item].append(float(row[item]))

    return result_dict1,result_dict2,result_dict3,result_dict4
    # return result_dict1,result_dict2,result_dict3
    # return result_dict1,result_dict2



def plot(x_label: str, x_value_list, y_label: str, y_value_list,figure_file_name: str):
    """Plot a figure."""
    fig, ax = plt.subplots()
    ax.plot(x_value_list[0], y_value_list[0],color='#2878b5',label='FedAsync')
    ax.plot(x_value_list[1], y_value_list[1],color='#9ac9db',label='FedAvg')
    ax.plot(x_value_list[2], y_value_list[2],color='#f8ac8c',label='FedBuff')
    ax.plot(x_value_list[3], y_value_list[3],color='#c82423',label='BSACS-FL')
    ax.set(xlabel=x_label, ylabel=y_label)
    # ax.set_title('不同时间片在',loc=CENTER)
    ax.legend()
    fig.savefig(figure_file_name)


def plot_figures_from_dict(result_csv_file_list,result_dir: str):
    """Plot figures with dictionary of results."""
    result_dict1,result_dict2,result_dict3,result_dict4 = read_csv_to_dict(result_csv_file_list)
    # result_dict1,result_dict2,result_dict3 = read_csv_to_dict(result_csv_file_list)
    # result_dict1,result_dict2 = read_csv_to_dict(result_csv_file_list)

    # plot_pairs = Config().results.plot
    plot_pairs = ['round&accuracy','round_time&accuracy']

    for pairs in plot_pairs:
        figure_file_name = result_dir + pairs + '.pdf'
        pair = [x.strip() for x in pairs.split('&')]
        x_y_labels: List = []
        x_y_values1: Dict[str, List] = {}
        x_y_values2: Dict[str, List] = {}
        x_y_values3: Dict[str, List] = {}
        x_y_values4: Dict[str, List] = {}
        for item in pair:
            label = {
                'global_round': 'Global training round',
                'round': 'Training round',
                'local_epoch_num': 'Local epochs',
                'accuracy': 'Accuracy (%)',
                'training_time': 'Training time (s)',
                'round_time': 'Round time (s)',
                'edge_agg_num': 'Aggregation rounds on edge servers'
            }[item]
            x_y_labels.append(label)
            x_y_values1[label] = result_dict1[item]
            x_y_values2[label] = result_dict2[item]
            x_y_values3[label] = result_dict3[item]
            x_y_values4[label] = result_dict4[item]


        x_label = x_y_labels[0]
        y_label = x_y_labels[1]

        x_value1 = x_y_values1[x_label]
        x_value2 = x_y_values2[x_label]
        x_value3 = x_y_values3[x_label]
        x_value4 = x_y_values4[x_label]
        x_value_list=[x_value1,x_value2,x_value3,x_value4]
        # x_value_list=[x_value1,x_value2,x_value3]
        # x_value_list=[x_value1,x_value2]
        
        y_value1 = x_y_values1[y_label]
        y_value2 = x_y_values2[y_label]
        y_value3 = x_y_values3[y_label]
        y_value4 = x_y_values4[y_label]
        y_value_list=[y_value1,y_value2,y_value3,y_value4]
        # y_value_list=[y_value1,y_value2,y_value3]
        # y_value_list=[y_value1,y_value2]

        plot(x_label, x_value_list, y_label, y_value_list,figure_file_name)


def main():
    """Plotting figures from the run-time results."""


    result_csv_file1 =  './exp3_performance_on_MNIST/noniid/FedAsync_MNIST_noniid.csv'
    result_csv_file2 =  './exp3_performance_on_MNIST/noniid/FedAvg_MNIST_noniid.csv'
    result_csv_file3 =  './exp3_performance_on_MNIST/noniid/FedBuff_MNIST_noniid.csv'
    result_csv_file4 =  './exp3_performance_on_MNIST/noniid/BSACS_MNIST_noniid.csv'

    result_csv_file_list=[result_csv_file1,result_csv_file2,result_csv_file3,result_csv_file4]
    # result_csv_file_list=[result_csv_file1,result_csv_file2,result_csv_file3]
    # result_csv_file_list=[result_csv_file1,result_csv_file2]
    result_dir = './'
    print("Plotting success.")
    plot_figures_from_dict(result_csv_file_list,result_dir)



if __name__ == "__main__":
    main()

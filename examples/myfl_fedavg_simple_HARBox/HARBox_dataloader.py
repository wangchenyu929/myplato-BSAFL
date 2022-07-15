import logging
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class HARBox(Dataset):

	class_set = ['Call','Hop','typing','Walk','Wave']
	label = [0,1,2,3,4]
	NUM_OF_CLASS = 5
	DIMENSION_OF_FEATURE = 900

	def __init__(self, train = True, client_id = None):

		x_coll, y_coll, dimension, num_of_class = self.load_data(client_id)
		if client_id == 0:
			x_train,x_test,y_train,y_test = self.generate_data(1, x_coll, y_coll)
		else:
			x_train,x_test,y_train,y_test = self.generate_data(0.1, x_coll, y_coll)
		if train:
			self.x_data = x_train.reshape([-1,1,30,30])
			self.y_data = y_train
		else:
			self.x_data = x_test.reshape([-1,1,30,30])
			self.y_data = y_test

	
	def __getitem__(self, index):
		return self.x_data[index],self.y_data[index]

	def __len__(self):
		return len(self.y_data)



	def load_data(self,user_id):

		# dataset append and split

		coll_class = []
		coll_label = []

		total_class = 0

		for class_id in range(self.NUM_OF_CLASS):
		
			read_path = './data/large_scale_HARBox_50clients/' +  str(user_id) + '/' + str(self.class_set[class_id]) + '_train' + '.txt'

			if os.path.exists(read_path):

				temp_original_data = np.loadtxt(read_path)
				temp_reshape = temp_original_data.reshape(-1, 100, 10)
				temp_coll = temp_reshape[:, :, 1:10].reshape(-1, self.DIMENSION_OF_FEATURE)
				count_img = temp_coll.shape[0]
				temp_label = class_id * np.ones(count_img)

				# print("temp_original_data:",temp_original_data.shape)
				# print("temp_coll:",temp_coll.shape)

				coll_class.extend(temp_coll)
				coll_label.extend(temp_label)

				total_class += 1

		coll_class = np.array(coll_class)
		coll_label = np.array(coll_label)

		# print("coll_class:",coll_class.shape)
		# print("coll_label:",coll_label.shape)

		return coll_class, coll_label, self.DIMENSION_OF_FEATURE, total_class


	def generate_data(self,test_percent, x_coll, y_coll):

		x_train,x_test,y_train,y_test = \
		train_test_split(x_coll,y_coll,test_size = test_percent,random_state = 0)

		return x_train,x_test,y_train,y_test

	def count_analysis(self,y):

		count_class = np.zeros(self.NUM_OF_CLASS)

		for class_id in range(self.NUM_OF_CLASS):
			count_class[class_id] = np.sum( y == class_id )

		return count_class


# my_dataset = HARBox(train = True, client_id=0)
# train_loader = DataLoader(dataset=my_dataset,
# 						batch_size=32,
# 						shuffle=False)
# print("count x_train",count_analysis(x_train))
# print("count x_test",count_analysis(x_test))
# print("count y_train",count_analysis(y_train))
# print("count y_test",count_analysis(y_test))
# my_dataset.count_analysis()
# for output_examples,output_labels in train_loader:
# 	print("output_examples")
# 	print(output_examples)
# 	print("output_labels")
# 	print(output_labels)
# print('train_loader',train_loader)
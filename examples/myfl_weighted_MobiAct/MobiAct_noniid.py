import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MobiAct(Dataset):


	class_set = ['STD','WAL','JOG','JUM','STU','STN','SCH','CSO','CSI','Fall']
	class_sample_num = [36,36,36,36,36,36,36,36,36,36]
	label = [0,1,2,3,4,5,6,7,8,9]
	NUM_OF_CLASS = 10


	def __init__(self, train = True, client_id = None):

		x_coll, y_coll = self.load_data(client_id)
		x_train,x_test,y_train,y_test = self.generate_data(1, x_coll, y_coll)
		if train:
			self.x_data = x_train
			self.y_data = y_train
		else:
			self.x_data = x_test
			self.y_data = y_test

	
	def __getitem__(self, index):
		return self.x_data[index],self.y_data[index]

	def __len__(self):
		return len(self.y_data)


	def load_data(self,client_id):
		# 用来存放class和label
		coll_class = np.zeros((144,1,30,30))
		coll_label = np.zeros((144))
		client_class_set = []
		if client_id in [1,6,11,16,21,26,31,36,41,46]:
			client_class_set = [0,1,2,3]
		elif client_id in [2,7,12,17,22,27,32,37,42,47]:
			client_class_set = [2,3,4,5]
		elif client_id in [3,8,13,18,23,28,33,38,43,48]:
			client_class_set = [4,5,6,7]	
		elif client_id in [4,9,14,19,24,29,34,39,44,49]:
			client_class_set = [6,7,8,9]
		elif client_id in [5,10,15,20,25,30,35,40,45,50]:
			client_class_set = [0,1,8,9]
		# client_class_set = [int((client_id-1)/10) *2,int((client_id-1)/10)*2+1]
		# print("client id:",client_id,client_class_set)
		
		sample_start = 0
		for class_id in client_class_set:
			read_path = './data/Annotated_Data/'+ str(self.class_set[class_id])+'/' + str(self.class_set[class_id]) + '_' + str(client_id) + '_1_annotated.csv'

			# print(read_path)

			temp_original_data = pd.read_csv(read_path)
			# coll_class[]开始的index
			# sample_start = sum(class_sample_num[:class_id])-class_sample_num[class_id]
			# sample_start = sum(self.class_sample_num[:class_id])
			# print('sample_start',sample_start)

			one_file_index = 0
			for first_index in range(sample_start,sample_start+self.class_sample_num[class_id]):
				temp_original =  np.array(temp_original_data.iloc[list(range(one_file_index,one_file_index+150)),[2,3,4,5,6,7]]).reshape(30,30)
				
				one_file_index += 150

				coll_class[first_index][0] = temp_original
				coll_label[first_index] = class_id

			sample_start = sample_start+self.class_sample_num[class_id]
			# print(coll_label)

		return coll_class,coll_label


	def generate_data(self,test_percent, x_coll, y_coll):

		x_train,x_test,y_train,y_test = \
		train_test_split(x_coll,y_coll,test_size = test_percent,random_state = 0)

		return x_train,x_test,y_train,y_test

	def count_analysis(self,y):

		count_class = np.zeros(self.NUM_OF_CLASS)

		for class_id in range(self.NUM_OF_CLASS):
			count_class[class_id] = np.sum( y == class_id )

		return count_class


# my_dataset = MobiAct(train = True, client_id=1)
# train_loader = DataLoader(dataset=my_dataset,
# 						batch_size=32,
# 						shuffle=False)
# for output_examples,output_labels in train_loader:
# 	print("output_examples")
# 	print(output_examples)
# 	print("output_labels")
# 	print(output_labels)
# print('train_loader',train_loader)


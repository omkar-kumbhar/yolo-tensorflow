import numpy as np 
import pandas as pd 
import os
import cv2

def csv_to_numpy(csv_path):
	assert os.path.exists(csv_path),"please provide correct path"
	data = pd.read_csv(csv_path)
	mnist = True
	if mnist:
		assert 'label' in data.columns
		y = np.array(data['label'])
		X = np.array(data.drop(['label'],axis=1))
	return X,y

class DataLoader:
	def __init__(self,input_):
		self.input_path = input_
		self.X_train = None
		self.y_train = None
		if self.input_path.endswith('.csv'):
			self.X, self.y = csv_to_numpy(self.input_path)
	
	def preprocess_mnist(self):
		self.X_train = self.X.reshape(self.X.shape[0],28,28,1).astype('float32') 
		# self.X_train = reshape(self.X_train)
		self.y_train = self.y.astype('int')

		self.X_train /= 255.
		return self.X_train,self.y_train


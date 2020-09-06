
import numpy as np
import os
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.backend import clear_session
def classifier(in_shape=20):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(10, input_shape = (in_shape,), activation='relu'))
	model.add(tf.keras.layers.Dense(10, activation='relu'))
	model.add(tf.keras.layers.Dense(2, activation='softmax'))
	model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])	
	return model


def train_classifier(x,y,x_val=None,y_val=None,n_epochs=100):
	# Normal model building
	model_main = classifier()
	if x_val is None or y_val is None:
		history = model_main.fit(x, y, epochs = n_epochs)
	else:
		history = model_main.fit(x, y,validation_data=(x_val,y_val), epochs = n_epochs)
		# print("************training evaluation**********")
		# print(model_main.evaluate(x, y,verbose=0))
		# print('***********done**************')
	return model_main, history

# preprocessing start

scaler = StandardScaler()

df = pd.read_csv('bank-additional-full.csv',header=None,sep=';')
	# df2 = pd.read_csv('trial.csv')
df = shuffle(df).reset_index(drop=True)

scaler = StandardScaler()
	# scaler = MinMaxScaler()

LABEL = df.columns[-1]
df.dropna()
DF = df


categorical=[2,3,4,5,6,7,8,9,10,15,21]

for i in categorical:  
	print(i)
	encoder = LabelEncoder()
	encoder.fit(df[i-1])
	df[i-1] = encoder.transform(df[i-1])



x = df.drop(LABEL, axis=1)


c = scaler.fit_transform(x)
x = pd.DataFrame(c)
X = x
Y = df[LABEL]
COLUMNS = len(X.columns)
print(COLUMNS)

# preprocessing completed
def run(CLIENTS=10):
	clients=CLIENTS
	IID = True
	Epochs=100
	DIRNAME='Federated_clients_'+str(clients)
	if IID:
		DIRNAME= DIRNAME+'_IID'
	else:
		DIRNAME= DIRNAME+'_Non_IID'
	trainX, testX, trainY, testY = train_test_split(np.array(X), np.array(Y), test_size = 0.2, random_state = 0)
	x = trainX
	y= trainY

	node_data_set = []
	for i in range(0,clients):
		print('client ',i)
		if IID:
			x_temp = x[int((i*len(x)/clients)):int((i+1)*len(x)/clients)]
			y_temp = y[int(i*len(x)/clients):int((i+1)*len(x)/clients)]
		else:
			x_temp = x[y==i]
			y_temp = y[y==i]
			break

		n_samples = len(y_temp)
		dataset = x_temp
		node_data_set.append((dataset,y_temp))
	training_accuracies=[]
	global_model = classifier()
	current_model_same=classifier()
	new_weights = list()
	for index in range(0,Epochs):
		#global_model.set_weights(new_weights)
		# try:
			
		# except:
		# 	print('an exception occured')
		weights=[]
		print('communication round ',index+1)
		for i in range(0,clients):
			print('Running data on node ',i+1)
			x_curr, y_curr = node_data_set[i]
			current_model = current_model_same
			current_model.set_weights(global_model.get_weights())
			current_model.fit(x_curr, y_curr, epochs = 1)
			weights.append(current_model.get_weights())
		
		new_weights.clear()
		
		for weights_list_tuple in zip(*weights):
			new_weights.append(
				np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))

		global_model.set_weights(new_weights)
		training_accuracy = global_model.evaluate(trainX,y)[1]
		print(training_accuracy)
		training_accuracies.append(training_accuracy)

	print('Complete')
	results={}
	results['federated global model testing accuracy'] = global_model.evaluate(testX,testY)[1]*100

	results_dir = DIRNAME
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	with open(results_dir+'/results.txt', 'w') as f:
		print(training_accuracies,file=f)
		print('printing other results',file=f)
		print(results,file=f)


CLIENTS=[10,1,2,25,50]
for index in CLIENTS:
	run(CLIENTS=index)
	clear_session()
	print('****************part completed****************')

from gan_utils import *
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.datasets.fashion_mnist import load_data
def classifier(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

def run(CLIENTS=10,iid=True):
	# size of the latent space
	clients = CLIENTS
	Epochs=100
	IID = iid
	DIRNAME='Federated_clients_'+str(clients)
	if IID:
		DIRNAME= DIRNAME+'_IID'
	else:
		DIRNAME= DIRNAME+'_Non_IID'
	(trainX, trainY), (testX, testY) = load_data()
	x = trainX
	y= trainY
	[testX,_] = load_real_samples(testX, testY)
	[trainX,_] = load_real_samples(trainX, trainY)
	node_data_set = []
	for i in range(0,clients):
		print('client ',i)
		if IID:
			x_temp = x[int((i*len(x)/clients)):int((i+1)*len(x)/clients)]
			y_temp = y[int(i*len(x)/clients):int((i+1)*len(x)/clients)]
		else:
			x_temp = x[y==i]
			y_temp = y[y==i]
		n_samples = len(y_temp)
		[dataset,_] = load_real_samples(x_temp, y_temp)
		node_data_set.append((dataset,y_temp))
	training_accuracies=[]
	global_model = classifier()

	for index in range(0,Epochs):
		models=[]
		print('communication round ',index+1)
		for i in range(0,clients):
			print('Running data on node ',i+1)
			x_curr, y_curr = node_data_set[i]
			current_model = classifier()
			current_model.set_weights(global_model.get_weights())
			
			current_model.fit(x_curr, y_curr, epochs = 1)
			models.append(current_model)
		weights = [model.get_weights() for model in models]
		new_weights = list()

		for weights_list_tuple in zip(*weights):
			new_weights.append(
				np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))

		global_model.set_weights(new_weights)
		training_accuracy = global_model.evaluate(trainX,y)[1]
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



run(CLIENTS=10,iid = False)
print('non-iid completed')
CLIENTS=[1,2,25,50]
for index in CLIENTS:
	run(CLIENTS=index)
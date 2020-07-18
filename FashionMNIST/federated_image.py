from gan_utils import *
import numpy as np
import os
from matplotlib import pyplot as plt

from keras.datasets.cifar10 import load_data
def classifier(in_shape=(32,32,3)):

	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
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
	
# size of the latent space
latent_dim = 100
clients = 10
Epochs=100
(trainX, trainY), (testX, testY) = load_data()
x = trainX
y= trainY
testX = load_real_samples(testX)
trainX = load_real_samples(trainX)
node_data_set = []
for i in range(0,clients):
	print('client ',i)
	x_temp = x[int((i*len(x)/clients)):int((i+1)*len(x)/clients)]
	y_temp = y[int(i*len(x)/clients):int((i+1)*len(x)/clients)]
	n_samples = len(y_temp)
	dataset = load_real_samples(x_temp)
	node_data_set.append((dataset,y_temp))
training_accuracies=[]
global_model = classifier()

for i in range(0,Epochs):
	models=[]
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

results_dir = 'federated_google_results'
if not os.path.isdir(results_dir):
	os.makedirs(results_dir)

with open(results_dir+'/results.txt', 'w') as f:
	print(training_accuracies,file=f)
	print('printing other results',file=f)
	print(results,file=f)


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
from keras.models import load_model
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
	
# size of the latent space
latent_dim = 100
clients = 10
(trainX, trainY), (testX, testY) = load_data()
x = trainX
y= trainY
[testX,_] = load_real_samples(testX, testY)
[trainX,_] = load_real_samples(trainX, trainY)
#combined_datset_x = np.empty_like(testX[0])
#combined_datset_y = np.empty_like(y)
combined_datset_x = []
combined_datset_y = []
for i in range(0,clients):
	print('client ',i)
	x_temp = x[int((i*len(x)/clients)):int((i+1)*len(x)/clients)]
	y_temp = y[int(i*len(x)/clients):int((i+1)*len(x)/clients)]
	n_samples = len(y_temp)
	# generate fake samples at the server
	path = 'generator_models_client_'+str(i+1)+'/generator_model_300.h5'
	g_model = load_model(path)
	[x_fake, labels], _ = generate_fake_samples(g_model, latent_dim, n_samples)
	y_fake = labels.reshape(-1,1)
	print(y_fake.shape)
	# store the combined datset
	if i==0:
		combined_datset_x = x_fake
		combined_datset_y = y_fake
	else:	
		combined_datset_x = np.vstack((combined_datset_x,x_fake))
		combined_datset_y = np.vstack((combined_datset_y,y_fake))

print('Complete')

combined_model, combined_model_history = train_classifier(combined_datset_x,combined_datset_y,x_val=trainX,y_val=trainY,n_epochs=100)

results = {}


results['FGAN testing accuracy(original_data)'] = combined_model.evaluate(testX,testY,verbose=0)[1]*100

if not os.path.isdir('testresults'):
	os.makedirs('testresults')

results_save_path = 'testresults/gan_distributed_results.txt'

with open(results_save_path, 'w') as f:

	print(combined_model_history.history['val_accuracy'],file=f)
	print('printing other results',file=f)
	print(results,file=f)
	print(combined_model_history.history['accuracy'],file=f)

x_axis= x_axis = [i for i in range(1,len(combined_model_history.history['accuracy'])+1)]
plt.plot(x_axis,combined_model_history.history['val_accuracy'], label='TestFGAN')
plt.plot(x_axis,combined_model_history.history['accuracy'],label='TrainFGAN')

plt.title('models accuracy comparision')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')

name = 'testresults/all_approaches_accuracy_vs_epochs'
plt.savefig(name+'.eps')
plt.savefig(name+'.png')
plt.clf()

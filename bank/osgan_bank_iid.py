from gan_utils import *
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

def run(CLIENTS=10, LATENT_DIM=100):
	# size of the latent space
	latent_dim = LATENT_DIM
	clients = CLIENTS

	DIRNAME='iid_clients_'+str(clients)
	trainX, testX, trainY, testY = train_test_split(np.array(X), np.array(Y), test_size = 0.2, random_state = 0)
	x = trainX
	y= trainY
	combined_datset_x = []
	combined_datset_y = []
	for i in range(0,clients):

		print('client ',i)
		x_temp = x[int((i*len(x)/clients)):int((i+1)*len(x)/clients)]
		y_temp = y[int(i*len(x)/clients):int((i+1)*len(x)/clients)]
		n_samples = len(y_temp)


		d_model = define_discriminator()
		# create the generatorg
		g_model = define_generator(latent_dim)
		# create the gan
		gan_model = define_gan(g_model, d_model)
		# train gan model
		train(g_model, d_model, gan_model, x_temp, latent_dim,n_epochs=100,client=i+1)

		image_model, _ = train_classifier(x_temp,y_temp,n_epochs=100)
		# generate fake samples at the server
		x_fake, _ = generate_fake_samples(g_model, latent_dim, n_samples)
		# classify fake samples
		y_fake = image_model.predict_classes(x_fake)
		y_fake = y_fake.reshape(-1,1)

		# store the combined datset
		if i==0:
			combined_datset_x = x_fake
			combined_datset_y = y_fake
		else:	
			combined_datset_x = np.vstack((combined_datset_x,x_fake))
			combined_datset_y = np.vstack((combined_datset_y,y_fake))



	print('Complete')

	central_model, central_model_history = train_classifier(trainX,trainY,n_epochs=100)
	combined_model, combined_model_history = train_classifier(combined_datset_x,combined_datset_y,x_val=trainX,y_val=trainY,n_epochs=100)

	results = {}
	results['OSGAN testing accuracy(original_data)'] = combined_model.evaluate(testX,testY,verbose=0)[1]*100
	results['Central model testing accuracy'] = central_model.evaluate(testX, testY, verbose=0)[1]*100


	dirname=DIRNAME+'/results'
	if not os.path.isdir(dirname):
		os.makedirs(dirname)
	results_save_path = dirname+'/gan_distributed_results.txt'

	with open(results_save_path, 'w') as f:

		print(combined_model_history.history['val_accuracy'],file=f)
		print('printing other results',file=f)
		print(results,file=f)
		print('******central model****',file=f)
		print(central_model_history.history['accuracy'],file=f)

	x_axis= x_axis = [i for i in range(1,len(central_model_history.history['accuracy'])+1)]
	plt.plot(x_axis,combined_model_history.history['val_accuracy'], label='OSGAN')
	plt.plot(x_axis,central_model_history.history['accuracy'],label='Central Learning')

	plt.title('models accuracy comparision')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(loc='lower right')

	name = dirname+'/all_approaches_accuracy_vs_epochs'
	plt.savefig(name+'.eps')
	plt.savefig(name+'.png')
	plt.clf()



CLIENTS=[10,1,2,25,50]
for index in CLIENTS:
	run(CLIENTS=index)
	clear_session()
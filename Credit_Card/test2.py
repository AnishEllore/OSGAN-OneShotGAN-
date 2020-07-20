from gan_u import *
import numpy as np
import os
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

def classifier(in_shape=23):
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

xls_file = pd.ExcelFile('default of credit card clients.xls')
df = xls_file.parse('Data')
new_header = df.iloc[0]
new_header[-1] = 'Y'
df = df[1:]
df.columns = new_header
df = shuffle(df).reset_index(drop=True)
df = df[df['Y']!='default payment next month']
X = df.drop(labels=['ID','Y'],axis=1).astype(np.float32)
X_Distributions = X
# if preprocessing is needed
# c = scaler.fit_transform(X)
# X = pd.DataFrame(c)

latent_dim = 50

Y = df['Y']
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

c = scaler.fit_transform(X)
X = pd.DataFrame(c)

X.columns = X_Distributions.columns

trainX, testX, trainY, testY = train_test_split(np.array(X), np.array(Y), test_size = 0.2, random_state = 0)

d_model = define_discriminator()
	# create the generator
g_model = define_generator(latent_dim)
	# create the gan
gan_model = define_gan(g_model, d_model)
	# load datset

i=0
n_samples = len(trainX)
	# train gan model
train(g_model, d_model, gan_model, trainX, latent_dim,n_epochs=100,client=i+1)

image_model, _ = train_classifier(trainX,trainY,n_epochs=100)
	# generate fake samples at the server
x_fake, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# classify fake samples
y_fake = image_model.predict_classes(x_fake)
y_fake = y_fake.reshape(-1,1)

	# generate fake samples at the server



print('Complete')

central_model, central_model_history = train_classifier(trainX,trainY,n_epochs=100)
combined_model, combined_model_history = train_classifier(x_fake,y_fake,x_val=trainX,y_val=trainY,n_epochs=100)

results = {}
results['FGAN testing accuracy(original_data)'] = combined_model.evaluate(testX,testY,verbose=0)[1]*100
results['Central model testing accuracy'] = central_model.evaluate(testX, testY, verbose=0)[1]*100


if not os.path.isdir('ganresults'):
	os.makedirs('ganresults')

results_save_path = 'ganresults/gan_distributed_results.txt'

with open(results_save_path, 'w') as f:

	print(combined_model_history.history['val_accuracy'],file=f)
	print('printing other results',file=f)
	print(results,file=f)
	print('******central model****')
	print(central_model_history.history['accuracy'],file=f)

x_axis= x_axis = [i for i in range(1,len(central_model_history.history['accuracy'])+1)]
plt.plot(x_axis,combined_model_history.history['val_accuracy'], label='FGAN')
plt.plot(x_axis,central_model_history.history['accuracy'],label='Central')

plt.title('models accuracy comparision')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')

name = 'ganresults/all_approaches_accuracy_vs_epochs'
plt.savefig(name+'.eps')
plt.savefig(name+'.png')
plt.clf()

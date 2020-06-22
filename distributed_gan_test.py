from gan_test import *
import numpy as np
import os
def classifier(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(10, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def train_classifier(x,y,x_val=None,y_val=None,n_epochs=100):
	# Normal model building
	model_main = classifier()
	if x_val is None or y_val is None:
		history = model_main.fit(x, y, epochs = n_epochs,verbose=0)
	else:
		history = model_main.fit(x, y,validation_data=(x_val,y_val), epochs = n_epochs, verbose=0)
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
testX = load_real_samples(testX)
trainX = load_real_samples(trainX)
combined_datset_x = np.empty_like(testX[0])
combined_datset_y = np.empty_like(y)
for i in range(0,clients):
	x_temp = x[int((i*len(x)/clients)):int((i+1)*len(x)/clients)]

	y_temp = y[int(i*len(x)/clients):int((i+1)*len(x)/clients)]


	n_samples = len(y_temp)
	# create the discriminator
	d_model = define_discriminator()
	# create the generator
	g_model = define_generator(latent_dim)
	# create the gan
	gan_model = define_gan(g_model, d_model)
	# load datset
	dataset = load_real_samples(x_temp)
	# train image model
	image_model, _ = train_classifier(dataset,y_temp,n_epochs=2)

	# train gan model
	train(g_model, d_model, gan_model, dataset, latent_dim,n_epochs=2)


	# generate fake samples at the server
	x_fake, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# classify fake samples
	y_fake = image_model.predict(x_fake)
	# store the combined datset
	combined_datset_x = np.append(combined_datset_x,x_fake,axis=0)
	combined_datset_y = np.append(combined_datset_y,y_fake,axis=0)

combined_model, combined_model_history = train_classifier(combined_datset_x,combined_datset_y,x_val=trainX,y_val=trainY,n_epochs=2)

central_model, central_model_history = train_classifier(trainX,trainY,n_epochs=2)
results = {}


results['Fusion testing accuracy(original_data)'] = combined_model.evaluate(testX,testY,verbose=0)[1]*100
results['Central model testing accuracy'] = central_model.evaluate(testX, testY, verbose=0)[1]*100


if not os.path.isdir('results'):
	os.makedirs('results')

results_save_path = 'results/gan_distributed_results.txt'

with open(results_save_path, 'w') as f:

	print(combined_model_history.history['val_accuracy'],file=f)
	print('printing other results',file=f)
	print(results,file=f)
	print('******central model****')
	print(central_model_history.history['accuracy'],file=f)
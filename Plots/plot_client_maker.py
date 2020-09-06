from matplotlib import pyplot as plt
import os


# MNIST
# gan = [97.89999723434448, 97.83999919891357, 96.92000150680542, 94.73999738693237, 92.69000291824341]
# federated = [98.7500011920929, 98.71000051498413, 98.05999994277954, 97.33999967575073, 95.57999968528748]

#Adult
# gan = [83.52395296096802, 83.55710506439209, 83.44107270240784, 82.79463052749634, 81.76695108413696]
# federated=[84.56820845603943, 84.71738696098328, 83.95491242408752, 83.29189419746399, 82.1647584438324]

#Bank
gan=[88.63801956176758, 88.63801956176758, 88.63801956176758, 91.3328468799591, 91.11434817314148]
federated = [91.66059494018555, 91.49065017700195, 91.46637320518494, 91.24787449836731, 90.6894862651825]
x_axis = [1,2,10,25,50]
plt.plot(x_axis,gan, label='OSGAN', color='red', marker='o')
#plt.plot(x_axis,central,label='Central Learning')
plt.plot(x_axis,federated,label='Federated Learning',color='green', marker='o')

#plt.title('models accuracy comparision')
plt.ylabel('Testing accuracy')
plt.xlabel('Clients')
plt.legend(loc='upper right')
plt.grid(True)

results_dir = 'Bank'
if not os.path.isdir(results_dir):
	os.makedirs(results_dir)

name = results_dir+'/accuracy_vs_clients'
plt.savefig(name+'.eps')
plt.savefig(name+'.png')
plt.clf()

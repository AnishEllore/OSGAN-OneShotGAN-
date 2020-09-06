import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 2

# gender
# means_local = [96.07843160629272, 96.07843160629272, 96.07843160629272, 96.07843160629272, 100.0, 92.15686321258545, 92.15686321258545, 94.11764740943909, 92.15686321258545, 96.07843160629272]
# means_global = [98.03921580314636, 94.11764740943909, 90.19607901573181, 100.0, 98.03921580314636, 92.15686321258545, 94.11764740943909, 94.11764740943909, 96.07843160629272, 96.07843160629272]

#MNIST
means_local = [93.97000074386597, 82.9200029373169]
means_global = [92.4,80.37999868392944]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index, means_local, bar_width,
alpha=opacity,
label='OSGAN Non-IID')

rects2 = plt.bar(index + bar_width, means_global, bar_width,
alpha=opacity,
label='Federated Non-IID')

plt.xlabel('Dataset')
plt.ylabel('Testing accuracy')
plt.xticks(index + bar_width/4, ('MNIST', 'FMNIST'))
plt.ylim(75,100)
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig('Non-IID-bar.eps')
plt.savefig('Non-IID-bar.png')

plt.show()
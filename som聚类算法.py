from imp import reload

from minisom import MiniSom
import numpy as np
import pandas as pd
import sys

data = pd.read_csv('D:\FirefoxDownload\Deep-Learning-master\Pytorch-Seg\lesson-2/data1.txt',
                    names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel'], usecols=[0, 4],
                   sep='\t+', engine='python')
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Initialization and training
som_shape = (1,6)

som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.7,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(data, 600, verbose=True)

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index

#==============================================
# h = winner_coordinates[1]
# print('h.shape',h.shape)
# print('h[0]:',h[0])
# file_handle=open('./append.txt',mode='r')
# for iteri in h:
#     #print(iteri)
#     file_handle.write(str(iteri))
#     file_handle.write('\n')
# file_handle.close()
#==============================================

# reload(sys)
# #sys.setdefaultencoding('utf8')
# fp = open("./append.txt", "r")
# sample = fp.readlines()
# kkk = 0
# print(winner_coordinates)
# for line in sample:
#     sample_ = line.split('\n')
#
#     print('sample_',sample_[0])
#     winner_coordinates[1][kkk] = sample_[0]
#     kkk +=1
# print(winner_coordinates)


cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)



import matplotlib.pyplot as plt
#%matplotlib inline

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], s=5 , label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    print(centroid)
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=2, linewidths=9, color='k', label='centroid')
plt.legend()
plt.grid()
plt.savefig('./11_7.png')
plt.show()
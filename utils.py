import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

def pca(data):
    return PCA(n_components=2).fit_transform(data)

def mask(corruption_level ,size):
    mask = np.random.binomial(1, 1 - corruption_level, [size[0],size[1]])
    return mask

def add_noise(x , corruption_level ):
    x = x * mask(corruption_level , x.shape)
    return x

def droupout(X , ratio):
    return add_noise(X , ratio)
def de_sigmoid(X):
    return X * (1.0 - X)

def sigmoid(X):
    return 1.0 / (1.0+np.exp(-X))

def plot(X):
    plt.plot([i for i in range(len(X))], X)
    plt.show()

def one_hot(labels):
    return np.array([[1 if i==label else 0 for i in range(10)] for label in labels])

def softmax(features):
    tmp = []
    for feature in features:
        feature=feature-np.max(feature)
        tmp.append(np.exp(feature)/sum(np.exp(feature)))
    return np.array(tmp)

def normalize(inputs):
    tmp = []
    for inp in inputs:
        tmp.append((inp - np.min(inp))/(np.max(inp) - np.min(inp)))
    return np.array(tmp)

def relu_de(L):
    batch,hidden = L.shape
    for i in range(batch):
        for j in range(hidden):
            if L[i][j]>0:
                L[i][j] = 1
            else:
                L[i][j] = 0
    return L

def cal_loss(Y,Y_):
    return -np.sum(Y_*np.log(Y+1e-9))/Y.shape[0]

def cal_loss_mse(Y,Y_):
    return np.sum((Y-Y_)**2)

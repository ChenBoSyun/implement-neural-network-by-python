# coding: utf-8

import load_data
import numpy as np
from utils import *

class Model():
    def __init__(self , x_train, y_train , x_test, y_test):
        self.lr = 0.0001
        self.batch_size = 128
        self.epochs = 10
        self.num_train = x_train.shape[0]
        self.num_test = x_test.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.total_loss = []
        self.total_acc = []
        self.set_weight()

    def set_weight(self):
        self.W1 = np.random.normal(0, 0.1, (784,256))
        self.W2 = np.random.normal(0, 0.1, (256,10))
        self.b1 = np.array([0.01 for i in range(256)])
        self.b1 = self.b1.reshape([1,256])
        self.b2 = np.array([0.01 for i in range(10)])
        self.b2 = self.b2.reshape([1,10])

    def forward_propagation(self,x):
        # 正傳導
        X = x
        L1 = X@self.W1 + self.b1
        H1 = np.maximum(L1, 0)
        L2 = H1@self.W2 + self.b2
        Y = softmax(L2)
        return Y



    def update(self,x,y):
        # 正傳導
        X = x
        L1 = X@self.W1 + self.b1
        H1 = np.maximum(L1, 0)
        L2 = H1@self.W2 + self.b2
        Y = softmax(L2)
        Y_ = y

        de_o = (Y-Y_)
        self.W2 = self.W2 - self.lr * H1.T@de_o
        self.b2 = self.b2 - self.lr * np.ones([1,self.batch_size])@de_o
        de_h2 = de_o@self.W2.T*relu_de(L1)
        self.W1 = self.W1 - self.lr * X.T@de_h2
        self.b1 = self.b1 - self.lr * np.ones([1,self.batch_size])@de_h2

    def fit(self):
        for epoch in range(self.epochs):
            for i in range(int(self.num_train/self.batch_size)):
                self.update(normalize(self.x_train[i*self.batch_size:(i+1)*self.batch_size]) , one_hot(self.y_train[i*self.batch_size : (i+1)*self.batch_size]))
            self.evaluate()
        return self.total_loss,self.total_acc

    def evaluate(self):
        total = self.num_test
        loss = 0
        num_true = 0
        for i in range(int(self.num_test/self.batch_size)):
            y_predict = self.forward_propagation(normalize (self.x_test[i*self.batch_size : (i+1)*self.batch_size]) )
            loss += cal_loss(y_predict , one_hot(self.y_test[i*self.batch_size : (i+1)*self.batch_size]))
            result = np.argmax(y_predict, axis=1)
            num_true += np.sum(result == self.y_test[i*self.batch_size : (i+1)*self.batch_size])


        self.total_loss.append(loss)
        self.total_acc.append(num_true/total)
        print("accuracy:%f"%(num_true/total))
        print("loss:%f"%(loss))

if __name__ == '__main__':
    x_train, y_train = load_data.loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = load_data.loadMNIST(dataset="testing", path="MNIST_data")

    model = Model(x_train, y_train,x_test, y_test)
    total_loss,total_acc = model.fit()
    plot(total_loss)
    plot(total_acc)

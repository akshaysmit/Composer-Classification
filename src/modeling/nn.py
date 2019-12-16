import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from load_data import *
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


from load_data import read_and_clean

OUT_BASE = "./out/pca"


class NN_classifier(nn.Module):
    def __init__(self, num_input, layers, num_output):
        super().__init__()
        
        layers = [num_input] + layers + [num_output]

        self.fc_layers = []
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.add_module("l_{}".format(i), self.fc_layers[-1])
        
        self.dropout = nn.Dropout(0.0)
        self.output_layer = nn.Softmax(dim=1)
        print(len(self.fc_layers))
    
    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i in  [0,1]:
                x = self.dropout(x)
            x = F.relu(x)
    
        return self.output_layer(x)


class NN():

    def __init__(self, num_input, num_output, training_epochs):
        super().__init__()
        hidden  = [150]
        self.model = NN_classifier(num_input, layers=hidden, num_output=num_output)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.04, momentum=0.9, weight_decay=0.001)
        self.loss_function = nn.CrossEntropyLoss()

        self.training_epochs = training_epochs
        self.batch_size = 50
 

    def fit(self, X, y, stats = False,  dev_X = None, dev_y = None):
             
        n = X.shape[0]
        m = X.shape[1]

        batch_data = []
        max_i = int(np.floor(n / self.batch_size) + 1)

        for i in range(max_i):
            i1 = i * self.batch_size
            i2 = (i + 1) * self.batch_size if i != max_i - 1 else n - 1
            if i2 - i1 > 0:
                batch_data.append((X[i1:i2], y[i1:i2]))

        X_dev_var = Variable(torch.Tensor(dev_X)) if stats else None
        X_train_var = Variable(torch.Tensor(X))
  
        accs = []
        for i in range(self.training_epochs):
            loss = None
            for batch_id, (x_batch, y_batch) in enumerate(batch_data):
                X_var = Variable(torch.Tensor(x_batch))
                Y_var = Variable(torch.Tensor(y_batch)).long()
    
                y_pred = self.model(X_var)
                loss = self.loss_function(y_pred, Y_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if stats and i % 20 == 0:
                dev_acc = np.sum(dev_y == np.apply_along_axis(np.argmax, 1, self.model(X_dev_var).data.numpy()))/dev_y.shape[0]
                train_acc = np.sum(y == np.apply_along_axis(np.argmax, 1, self.model(X_train_var).data.numpy()))/y.shape[0]
                print("i = {} train accuracy = {:.3f}, dev accuracy = {:.3f}".format(i, train_acc, dev_acc))
                accs.append((train_acc, dev_acc))

    def predict(self, X):
        X_var = Variable(torch.Tensor(X))
        return self.model(X_var).data.numpy()

    
    def score(self, X, y):
        X_var = Variable(torch.Tensor(X))
        return np.sum(y == np.apply_along_axis(np.argmax, 1, self.model(X_var).data.numpy()))/y.shape[0]        

        


        #y1, y2 = [np.array(arr) for arr in zip(*accs)]
        #x = np.linspace(0, y1.shape[0] * 20, y1.shape[0])
#
        #plt.figure()
        #plt.plot(x, y1)
        #plt.plot(x, y2)
        #plt.savefig("{}/nn_train.png".format(OUT_BASE))
        #plt.show()




def main():
    num_labels = 6
    train_X, train_y, dev_X, dev_y, headers = read_and_clean(standardize=True, num_labels=num_labels, same_size=False)

    clf = NN(train_X.shape[1], num_labels, 600)
    clf.fit(train_X, train_y,stats=True, dev_X = dev_X, dev_y = dev_y)
    print(clf.score(dev_X, dev_y))





if __name__ == "__main__":

    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)
    torch.manual_seed(0)
    main()

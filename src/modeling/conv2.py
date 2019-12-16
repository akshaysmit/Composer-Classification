import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import csv 
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


from pca import save_confusion_m
from sklearn.metrics import confusion_matrix
from pca import get_y_labels

import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10

learning_rate = 0.0003



class ConvNet2(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(1,4), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(1,2)))

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(1,2)))

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)))
        

        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)))
        
  

        self.fc1 = nn.Linear(2*3*1024, 512)
        self.fc2 = nn.Linear(512, 96)
        self.fc3 = nn.Linear(96, num_classes)
        self.training = True
        
    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)

        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = self.layer6(out)
        #print(out.shape)
        out = self.layer7(out)

        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        
        out = F.dropout(F.relu(self.fc1(out)),p = 0.3,training=self.training)
        out = F.dropout(F.relu(self.fc2(out)),p = 0.3,training=self.training)
        out = self.fc3(out)
        #print(out.shape)
        return out
     


def move(data, dx, dy):
    new = np.zeros_like(data)

    
    new = np.roll(data, dy, axis=0)
    if dy >= 0:
        new[0:dy,:] = 0
    else:
        new[dy:,:] = 0
    new = np.roll(new, dx, axis=1)
    if dx >= 0:
        new[:,0:dx] = 0
    else:
        new[:,dx:] = 0
    return new




def augment_data_random(data,  k):
    new_data = [data]

    for i in range(k-1):
        dx = np.random.randint(-150, 150)
        dy = np.random.randint(-15, 15)
        new = move(data,dx, dy)

        z = np.random.randint(-50, 50)
        z = z + 64*12
        new[:,z-50:z+50] = 0
        new_data.append(new)
    return new_data  


def load_data(K):
    base = "../data_cleaning/"
    f = open("{}train_data.csv".format(base))
    reader = csv.reader(f)
    
    lines = [line for line in reader]
    random.shuffle(lines)
    counts = np.array([int(line[0]) for line in lines])
    filter_indices = np.argsort(np.bincount(counts))[-6:]
    lines = [line for line in lines if int(line[0]) in filter_indices]

    print(filter_indices)
    indices_map = {index : i for i,index in enumerate(filter_indices)}
    for i,_ in enumerate(lines):
        lines[i][0] = indices_map[int(lines[i][0])]
    
    print(np.bincount(counts))
    line_dev = [line for line in lines if line[2] == "1"]
    line_test = [line for line in lines if line[2] == "2"]
    lines = [line for line in lines if line[2] == "0"]

    batch_size = int(100 * K / 15.0)

    
    batch_data = []
    test_data = []
    

    N = len(lines) * K
    N_dev = len(line_dev)
    N_test = len(line_test)
    #use image size of 64 x (64 * 24)
    #X = np.zeros([N, 64, 64 * 24], dtype='uint8')
    X = np.empty(shape=(N,), dtype=object) #list of sparse matrices
    X_dev = np.zeros([N_dev, 64, 64 * 24], dtype='uint8')
    X_test = np.zeros([N_test, 64, 64 * 24], dtype='uint8')
    y = np.zeros(N)
    y_dev = np.zeros(N_dev)
    y_test = np.zeros(N_test)


    for j, line in enumerate(lines):
        if j % 100 == 0:
            print("Reading", j)
        composer = line[0]
        song = line[1]
        image_path = "{}/train_images/{}.png".format(base, song)
        img = np.asarray(Image.open(image_path).convert("L"))
        #for i in range(9):
        
        augmented = augment_data_random(img, K)
        for i, d in enumerate(augmented):
            data = d[32:96, 0: 24 * 64].astype(float)
            data = data / 128.0
            X[j*K + i] = csr_matrix(data)
            y[j*K + i] = composer
    
    
    shuffle_indices = np.arange(y.shape[0])
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]  
    y = y[shuffle_indices]      

    for j, line in enumerate(line_dev):
        composer = line[0]
        song = line[1]
        image_path = "{}/train_images/{}.png".format(base, song)
        img = np.asarray(Image.open(image_path).convert("L"))
        #for i in range(9):
        data = img[32:96, 0: 24 * 64]
        X_dev[j,:,:] = data
        y_dev[j] = composer
    X_dev = X_dev / 128.0
    X_dev = X_dev[:,None,:,:]

    for j, line in enumerate(line_test):
        composer = line[0]
        song = line[1]
        image_path = "{}/train_images/{}.png".format(base, song)
        img = np.asarray(Image.open(image_path).convert("L"))
        #for i in range(9):
        data = img[32:96, 0: 24 * 64]
        X_test[j,:,:] = data
        y_test[j] = composer
    X_test = X_test / 128.0
    test = X_test[:,None,:,:]

    max_i = int(np.floor(N / batch_size) + 1)

    for i in range(max_i):
        i1 = i * batch_size
        i2 = (i + 1) * batch_size if i != max_i - 1 else N - 1
        if i2 - i1 > 0:
            batch_data.append((X[i1:i2], y[i1:i2]))
    
    dev_data = (X_dev, y_dev)
    test_data = ( X_test, y_test)
    

    return (batch_data, dev_data, test_data)

OUT_BASE = "./out/CNN/"

def train_plot(K):
    batch_data,dev_data, test_data = load_data(K)

    model = ConvNet2(6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dev_images, dev_labels =  dev_data
    X_dev_var = Variable(torch.Tensor(dev_images)).to(device)

    train_accurcy = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print("epoch ", epoch)
        model.training = False
        dev_acc = np.sum(dev_labels == np.apply_along_axis(np.argmax, 1, model(X_dev_var).cpu().data.numpy()))/dev_labels.shape[0] 
        train_accurcy[epoch] = dev_acc
        model.training = True
        print('Accuracy = {}'.format(dev_acc))
        for i, (images, labels) in enumerate(batch_data):
            #images is a list of sparse matrices
            M = labels.shape[0]
            X = np.zeros([M , 1, 64, 64 * 24])
            for j, image in enumerate(images):
                X[j,:,:,:] = image.toarray()[None,:,:]
            shuffle_indices = np.arange(M)
            np.random.shuffle(shuffle_indices)
            X = X[shuffle_indices]
            labels = labels[shuffle_indices]
            X_var = Variable(torch.Tensor(X)).to(device)
            Y_var = Variable(torch.Tensor(labels)).long().to(device)

            output = model(X_var)
            loss = criterion(output, Y_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                model.training = False
                train_acc = np.sum(labels == np.apply_along_axis(np.argmax, 1, model(X_var).cpu().data.numpy()))/labels.shape[0] 
                model.training = True
                print("train acc", train_acc)
                #print(loss.item())
    return train_accurcy
    
def main():

    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)

    results = []
    K = [1,3,5,8,13,15,18]
    #K = [1,3,5]
    for k in K:
        res = train_plot(k)
        results.append(res)

    np.savetxt("{}/result.txt".format(OUT_BASE), results)

    x = np.linspace(0,num_epochs,num_epochs)
    plt.figure()
    plt.title("CNN accuracy")
    for i, k in enumerate(K):
        res = results[i]
        plt.plot(x, res, label="K = {}".format(k))
        plt.legend()
    
    plt.savefig("{}/different_k.png".format(OUT_BASE))
    plt.show()

    



if __name__ == "__main__":
    
    main()
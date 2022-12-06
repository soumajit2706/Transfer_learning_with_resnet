import pickle
import torch
import os

class train_tensor():
    def __init__(self,filenames): 
        self.count = 0
        self.datafile = {}
        for file in filenames:
            temp = self.unpickle(file)
            self.labels = temp[b'labels']
            self.data = temp[b'data']
            self.data_divider()

    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo,encoding='bytes')
        return dict

    def data_divider(self):
        temp = self.count 
        for i in range(self.count,self.count + len(self.labels)):
            if not os.path.exists('./dataset/cifar_train/'):
                os.mkdir('./dataset/cifar_train/')
            torch.save(torch.reshape(torch.from_numpy(self.data[i-temp]),(3,32,32)),'./dataset/cifar_train/' + str(self.count) + '.pt')
            self.datafile[self.count] = self.labels[i-temp]
            self.count += 1


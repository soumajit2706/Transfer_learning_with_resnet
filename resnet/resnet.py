import pickle
import torch
import os
import random


class train_tensor():
    def __init__(self,filenames): 
        self.count = 0
        self.train_count = 0
        self.valid_count = 0
        self.traindata = {}
        self.validdata = {}
        temp = [i for i in range(50000)]
        random.shuffle(temp)
        self.valid_ind = temp[:5000]
        for file in filenames:
            temp = self.unpickle(file)
            self.labels = temp[b'labels']
            self.data = temp[b'data']
            self.data_divider()

        pickle.dump(self.traindata,open('../dataset/cifar_train/train_dataset.pkl','wb'))
        pickle.dump(self.validdata,open('../dataset/cifar_valid/valid_dataset.pkl','wb'))

    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo,encoding='bytes')
        return dict

    def data_divider(self):
        temp = self.count 
        for i in range(self.count,self.count + len(self.labels)):
            if i not in self.valid_ind:
                if not os.path.exists('../dataset/cifar_train/'):
                    os.mkdir('../dataset/cifar_train/')
                torch.save(torch.reshape(torch.from_numpy(self.data[i-temp]),(3,32,32)),'../dataset/cifar_train/' + str(self.train_count) + '.pt')
                self.traindata[self.train_count] = self.labels[i-temp]
                self.count += 1
                self.train_count +=1 
            else:
                if not os.path.exists('../dataset/cifar_valid/'):
                    os.mkdir('../dataset/cifar_valid/')
                torch.save(torch.reshape(torch.from_numpy(self.data[i-temp]),(3,32,32)),'../dataset/cifar_valid/' + str(self.valid_count) + '.pt')
                self.validdata[self.valid_count] = self.labels[i-temp]
                self.count += 1
                self.valid_count +=1






import pickle
import torch
import os
import random
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T

class train_tensor():
    def __init__(self,filenames): 
        self.count = 0
        self.train_count = 0
        self.valid_count = 0
        self.flag_sum = torch.zeros([3,32,32],dtype=torch.float32) 
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
            print(self.flag_sum)
        self.mean = torch.mean(self.flag_sum/50000,dim=(1,2))
        self.std = torch.mean(self.flag_sum/50000,dim=(1,2))

        pickle.dump(self.mean,open('../dataset/cifar_train/train_mean.pkl','wb'))
        pickle.dump(self.std,open('../dataset/cifar_train/train_std.pkl','wb'))
        pickle.dump(self.traindata,open('../dataset/cifar_train/train_dataset.pkl','wb'))
        pickle.dump(self.validdata,open('../dataset/cifar_valid/valid_dataset.pkl','wb'))

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo,encoding='bytes')
        return dict

    def data_divider(self):
        temp = self.count 
        for i in range(self.count,self.count + len(self.labels)):
            if i not in self.valid_ind:
                if not os.path.exists('../dataset/cifar_train/'):
                    os.mkdir('../dataset/cifar_train/')
                flag = torch.reshape(torch.from_numpy(self.data[i-temp]),(3,32,32))
                self.flag_sum += flag 
                torch.save(flag,'../dataset/cifar_train/' + str(self.train_count) + '.pt')
                self.traindata[self.train_count] = self.labels[i-temp]
                self.count += 1
                self.train_count +=1 
            else:
                if not os.path.exists('../dataset/cifar_valid/'):
                    os.mkdir('../dataset/cifar_valid/')
                flag = torch.reshape(torch.from_numpy(self.data[i-temp]),(3,32,32))
                self.flag_sum += flag
                torch.save(flag,'../dataset/cifar_valid/' + str(self.valid_count) + '.pt')
                self.validdata[self.valid_count] = self.labels[i-temp]
                self.count += 1
                self.valid_count +=1

class test_tensor(train_tensor):
    def __init__(self,filename):
        self.testdata = {}
        self.flag_sum = torch.zeros([3,32,32],dtype=torch.float32)
        temp = train_tensor.unpickle(filename)
        self.labels = temp[b'labels']
        self.data = temp[b'data']
        self.data_divider()
        self.mean = torch.mean(self.flag_sum/10000,dim=(1,2))
        self.std = torch.mean(self.flag_sum/10000,dim=(1,2))

        pickle.dump(self.mean,open('../dataset/cifar_test/test_mean.pkl','wb'))
        pickle.dump(self.std,open('../dataset/cifar_test/test_std.pkl','wb'))
        pickle.dump(self.testdata,open('../dataset/cifar_test/test_dataset.pkl','wb'))

    def data_divider(self):
        for i in range(0,len(self.labels)):
            if not os.path.exists('../dataset/cifar_test/'):
                os.mkdir('../dataset/cifar_test/')
            flag  = torch.reshape(torch.from_numpy(self.data[i]),(3,32,32))
            self.flag_sum += flag
            torch.save(flag,'../dataset/cifar_test/' + str(i) + '.pt')
            self.testdata[i] = self.labels[i]






class Cumtomdataset(Dataset):
    def __init__(self,filename):
        super().__init__()
        self.dataset = pickle.load(open(filename,'rb'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        x = torch.load(str(idx) + '.pt')
        y = self.dataset[idx]
        return x,y






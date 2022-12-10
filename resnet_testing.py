from resnet import resnet 
import glob 
import numpy as np
import os 
import pickle

if not os.path.exists('./dataset/cifar_train/'):
    dataset = glob.glob('../dataset/cifar-10-batches-py/data_batch_*')
    data = resnet.train_tensor(dataset)

if not os.path.exists('./dataset/cifar_test/'):
    data = resnet.test_tensor('../dataset/cifar-10-batches-py/test_batch')

mean = pickle.load(open('../dataset/cifar_train/train_mean.pkl','rb'))
std = pickle.load(open('../dataset/cifar_train/train_std.pkl','rb'))

print(mean,std)

mean = pickle.load(open('../dataset/cifar_test/test_mean.pkl','rb'))
std = pickle.load(open('../dataset/cifar_test/test_std.pkl','rb'))

print(mean,std)

import random 
import numpy as np
import torchvision.transforms as T
import pickle
import torch
import matplotlib.pyplot as plt

dataset = pickle.load(open('../dataset/cifar_valid/valid_dataset.pkl','rb'))
lis = random.sample(range(0,5000),10)

for i in lis:
    filename = '../dataset/cifar_valid/' + str(i) + '.pt'
    tensor = torch.load(filename)
    transform = T.ToPILImage()
    img = transform(tensor)
    img.save(str(i) + '.png')
    print(str(i) + ' : ' + str(dataset[i]))

    




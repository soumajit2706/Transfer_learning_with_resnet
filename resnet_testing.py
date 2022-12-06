from resnet import resnet 
import glob 
import numpy as np

dataset = glob.glob('./dataset/cifar-10-batches-py/data_batch_*')

data = resnet.train_tensor(dataset)



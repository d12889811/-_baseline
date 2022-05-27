from model import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import utils
import argparse
from Dataset.myDataset import JAFFEDataset

#设置参数

BATCH_SIZE=32
model_path=os.path.join(os.getcwd(),'save_model','resNet50_DA_split_epoch_39.pkl')
device = torch.device("cuda")
test_dir='/home/ck+_crop'


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_data = JAFFEDataset(data_dir=test_dir, transform=test_transform)

test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,num_workers=6)

test_dataset_size = len(test_data)
print(test_dataset_size)

print('==> Loading model..')
#net=VGG('VGG19')
net = torchvision.models.resnet50()
net.load_state_dict(torch.load(model_path))
net.to(device)
print('==> Building model finish')

correct_val = 0.
net.eval()
with torch.no_grad():
    for j, data in enumerate(test_loader):
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = net(inputs)
        predicted = torch.argmax(outputs.data, dim=1)
	
        correct_val += (predicted == labels).sum().item()

    print("Test:\t Acc:{:.2%}".format( correct_val / test_dataset_size))
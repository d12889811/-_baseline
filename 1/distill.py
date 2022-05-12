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
import argparse
from Dataset.myDataset import JAFFEDataset
from torch.autograd import Variable
from tensorboardX import  SummaryWriter
from tqdm import tqdm
torch.backends.cudnn.benchmark=True

MAX_EPOCH = 10
BATCH_SIZE=2
LR = 0.01
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
log_interval = 10
val_interval = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

train_dir='/media/dong/Ventoy/dataset/jaffe/split/train'
valid_dir='/media/dong/Ventoy/dataset/jaffe/split/valid'
test_dir='/media/dong/Ventoy/dataset/jaffe/split/test'

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = JAFFEDataset(data_dir=train_dir, transform=train_transform)
valid_data = JAFFEDataset(data_dir=valid_dir, transform=valid_transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


CurrentNet = VGG('VGG19')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CurrentNet.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



epochs = 6 # 训练6轮
for epoch in range(epochs):
    CurrentNet.train()

    for data,targets in tqdm(train_loader):
        # 前向预测
        preds = CurrentNet(data)
        loss = criterion(preds,targets)

        # 反向传播，优化权重
        optimizer.zero_grad()  # 把梯度置为0
        loss.backward()
        optimizer.step()

    # 测试集上评估性能
    CurrentNet.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in valid_loader:
            preds = CurrentNet(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()

    print(("Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))

teacher_model = CurrentNet

CurrentNet= StudentNet

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CurrentNet.parameters(),lr=1e-4)

epochs = 3
# 训练集上训练权重
for epoch in range(epochs):
    CurrentNet.train()

    for data,targets in tqdm(train_loader):
        # 前向预测
        preds = CurrentNet(data)
        loss = criterion(preds,targets)

        # 反向传播，优化权重
        optimizer.zero_grad() # 把梯度置为0
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for x,y in  valid_loader:
            preds = CurrentNet(x)
            predictions = preds.max(1).indices
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            acc = (num_correct / num_samples).item()

    print(("Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))

student_model_scratch = CurrentNet

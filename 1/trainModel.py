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
from torch.autograd import Variable
#设置参数
MAX_EPOCH = 50
BATCH_SIZE=32
LR = 0.0002
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
log_interval = 1
val_interval = 0
learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 1 # 5
learning_rate_decay_rate = 0.8 # 0.9


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

train_dir='/home/fer_sub_DA'
valid_dir='/home/1/jaffe/split/valid'
test_dir='/home/1/jaffe/split/test'

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



train_dataset_size = len(train_data)
val_dataset_size = len(valid_data)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

print('==> Building model..')
net = VGG('VGG19')
net.cuda()
print('==> Building model finish')


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999),eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9


train_curve = list()
valid_curve = list()

iter_count = 0

# 构建 SummaryWriter
#writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")


for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = LR * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = LR
    print('learning_rate: %s' % str(current_lr))
    # 遍历 train_loader 取数据
    for i, data in enumerate(train_loader):
        iter_count += 1
        # forward
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item()/BATCH_SIZE)
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean /BATCH_SIZE, correct / total))
            loss_mean = 0.

        # 记录数据，保存于event file
        #writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
        #writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)

    # 每个epoch，记录梯度，权值
    #for name, param in net.named_parameters():
        #writer.add_histogram(name + '_grad', param.grad, epoch)
        #writer.add_histogram(name + '_data', param, epoch)

    #scheduler.step()  # 每个 epoch 更新学习率
    # 每个 epoch 计算验证集得准确率和loss
    # validate the model
    if val_interval != 0 and (epoch+1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                predicted = torch.argmax(outputs.data, dim=1)
                correct_val += (predicted == labels).sum().item()

                loss_val += loss.data.item()

            valid_curve.append(loss_val/val_dataset_size)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val / val_dataset_size, correct_val / val_dataset_size))
            # 记录数据，保存于event file
            #writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
            #writer.add_scalars("Accuracy", {"Valid": correct / total}, iter_count)

if not os.path.exists(os.path.join(os.getcwd(),'save_model')):
    os.mkdir(os.path.join(os.getcwd(),'save_model'))
torch.save(net.state_dict(), os.path.join(os.getcwd(),'save_model'))
train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
if len(valid_y)>0:
    plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.savefig('./cross_result_loss.jpg')

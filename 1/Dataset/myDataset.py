   
# -*- coding: utf-8 -*-
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np
import torch.optim as optim

fer_label={'anger':0,'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}

class JAFFEDataset(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        
    def __getitem__(self, index):
        # 通过 index 读取样本
        path_img, label = self.data_info[index]
        # 注意这里需要 convert('RGB')
        img = Image.open(path_img).convert('I')# 0~255
        img=np.asarray(img)
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        # 返回是样本和标签
        return img, label        
    
    
    def __len__(self):
        return len(self.data_info)
        
        
    @staticmethod
    def get_img_info(data_dir):
        data_info=list()
        
        for root, dirs, _ in os.walk(data_dir):
            # dirs =[7种表情分类]
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                #只读取jpg图片
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 图片的绝对路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 标签 0-6
                    label = fer_label[sub_dir]
                    # 保存在 data_info 变量中
                    data_info.append((path_img, int(label)))
        return data_info
    
    
class MyTransform(object):
    def __call__(self, img):
        img=torch.tensor(np.asarray(img))
        img=torch.squeeze(img)
        
        return img

        img=torch.tensor(np.asarray(img))
        img=torch.squeeze(img)
        
        return img

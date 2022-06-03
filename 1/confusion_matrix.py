
# -*- coding: utf-8 -*-
""" 
@Time    : 2021/11/18 0:33
@Author  : ONER
@FileName: plt_cm.py
@SoftWare: PyCharm
"""
 
#confusion_matrix
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
 
classes = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
confusion_matrix = np.array([(9,13,1,0,12,10,0),(4,49,0,4,2,0,0),(0,2,4,8,6,3,2),(0,0,0,69,0,0,0),(24,19,14,24,345,160,7),(3,5,2,0,5,13,0),(1,4,8,1,9,2,58)],dtype=np.int32)#输入特征矩阵
proportion=[]
for i in confusion_matrix:
    for j in i:
        temp=j/(np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
#print(proportion)
print("aaa")
pshow=[]
for i in proportion:
    pt="%.2f%%" % (i * 100)
    pshow.append(pt)
proportion=np.array(proportion).reshape(7,7)  # reshape(列的长度，行的长度)
pshow=np.array(pshow).reshape(7,7)
#print(pshow)
print("aaa")
config = {
    "font.family":'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
            # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
            # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
plt.title('confusion_matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes,fontsize=12)
plt.yticks(tick_marks, classes,fontsize=12)
print("aaa")
thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(7)] for i in range(7)],(confusion_matrix.size,2))
for i, j in iters:
    if(i==j):
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12,color='white',weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12,color='white')
    else:
        plt.text(j, i-0.12, format(confusion_matrix[i, j]),va='center',ha='center',fontsize=12)   #显示对应的数字
        plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=12)
 
plt.ylabel('True label',fontsize=16)
plt.xlabel('Predict label',fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'confusion_matrix.jpg'))
print("aaa")
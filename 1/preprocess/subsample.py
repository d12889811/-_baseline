import os
import random
import shutil
from random import sample

data_dir=os.path.join(os.getcwd(),'fer2013')
save_dir=os.path.join(os.getcwd(),'fer_subset')
sampleNum=500
expressionlabel=['anger','disgust','fear','happiness','sadness','surprise','neutral']

def createDirs():
    for x in expressionlabel:
        classPath=os.path.join(save_dir,x)
        if os.path.exists(classPath) is False:
            os.makedirs(classPath)





if __name__ == '__main__':
    createDirs()
    i=0
    for root , dirs, _ in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d[0] == '.']
    #dirs:[7 expressions]
        for sub_dir in dirs:
            imgs = os.listdir(os.path.join(root, sub_dir))
            # 取出 jpg 结尾的文件
            print(len(imgs))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            imgs=sample(imgs,sampleNum)
            for image in imgs:
                src_path=os.path.join(root,sub_dir,image)
                target_path=os.path.join(save_dir,sub_dir,image)
                shutil.copy(src_path, target_path)
                i+=1
                print("已完成: {} 张图片的处理".format(i))

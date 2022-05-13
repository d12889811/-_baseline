import os.path
from test22 import cropping

cfg = {
    'jaffe':['anger','disgust','fear','happiness','neutral','sadness','surprise'],
    'fer2013':['anger','disgust','fear','happiness','neutral','sadness','surprise']
}

def cropImage(setName,destName=None):
    i=0
    if destName is None:
        destName=setName+'_crop'
    destPath=os.path.join(os.getcwd(),destName)
    if os.path.exists(destPath) is False:
        os.makedirs(destName)
    inpath=os.path.join(os.getcwd(),setName)
    for x in cfg[setName]:
        ClassDestPath=os.path.join(destPath,x)
        if os.path.exists(ClassDestPath) is False:
            os.makedirs(ClassDestPath)
        classPath=os.path.join(inpath,x)
        files = os.listdir(classPath)
        files.sort()
        for image in files:
            cropping(os.path.join(classPath, image), os.path.join(ClassDestPath, image))
            i+=1
            print("已完成 ：{} 張圖片的處理".format(i))

if __name__ == '__main__':
    cropImage('fer2013')
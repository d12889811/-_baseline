import os
import numpy as np
import random as rd
import torchvision
import cv2

indir="C:\\Users\\12169\\Desktop\\course\\program\\facial_expression_recog\\dataset\\ck+"


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def crop(image, min_ratio=0.85, max_ratio=1.0):
    h, w = image.shape[:2]
    ratio = rd.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h*scale)
    new_w = int(w*scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[y:y+new_h, x:x+new_w, :]
    return image

def change(image):
    x,y = image.shape[:2]
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M,(y,x),borderValue=(255,255,255))
    return dst

def Horizontal(image):
    return cv2.flip(image,1,dst=None)

def Darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

def Brighter(image, percetage=1.1):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy

def SaltAndPepper(src,percetage):
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randR=np.random.randint(0,src.shape[0]-1)
        randG=np.random.randint(0,src.shape[1]-1)
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0:
            SP_NoiseImg[randR,randG,randB]=0
        else:
            SP_NoiseImg[randR,randG,randB]=255
    return SP_NoiseImg

def GaussianNoise(image,percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


def Blur(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    return blur


def AllData():
    root_path = os.path.join(os.getcwd(),'jaffe','original')
    save_loc = os.path.join(os.getcwd(),'jaffe_DA')
    makedir(save_loc)
    for root, dirs, files in os.walk(os.path.normpath(root_path)):
        for dir in dirs:
            makedir(os.path.join(save_loc,dir))
            images=os.listdir(os.path.join(root,dir))
            images = list(filter(lambda x: x.endswith('.jpg'), images))
            for image in images:
                imPath=os.path.join(root,dir,image)
                tarPath=os.path.join(save_loc,dir)
                ori=cv2.imread(imPath)

                DA_img=crop(ori)
                print(os.path.join(tarPath,image[:-4]+"_DA_crop.png"))
                cv2.imwrite(os.path.join(tarPath,image[:-4]+"_DA_crop.png"),DA_img)

def onepic(pic):
    ori = cv2.imread(pic)
    DA_img = GaussianNoise(ori,0.08)
    print(os.path.join(pic[:-4] + "_DA.jpg"))
    cv2.imwrite(os.path.join(pic[:-4] + "_DA.jpg"), DA_img)
if __name__ == "__main__":
    # TestOneDir()
    # TestOnePic()
    AllData()
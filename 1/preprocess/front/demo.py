import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
        # check for dlib saved weights for face landmark detection
        # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

this_path = os.path.dirname(os.path.abspath(__file__))
expressionlabel=['anger','disgust','fear','happiness','sadness','surprise','neutral']
test_dir=os.path.join(os.getcwd(),'test')
save_dir=os.path.join(os.getcwd(),'test_save')


def createDirs():
    for x in expressionlabel:
        classPath=os.path.join(save_dir,x)
        if os.path.exists(classPath) is False:
            os.makedirs(classPath)

def demo():
    i=0
    for root,dirs,images in os.walk(test_dir):
        for sub_dir in dirs:
            images = os.listdir(os.path.join(root, sub_dir))
            images = list(filter(lambda x: x.endswith('.jpg'), images))
            for image in images:
                check.check_dlib_landmark_weights()
                model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
                img = cv2.imread(os.path.join(test_dir,sub_dir,image), 1)
                lmarks = feature_detection.get_landmarks(img)
                if len(lmarks) !=0 :
                    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
                    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
                    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
                    cv2.imwrite(os.path.join(save_dir,sub_dir,"{}_front".format(image)),frontal_sym)
                    i+=1
                    print("已完成: {} 张图片的处理".format(i))
    


if __name__ == "__main__":
    createDirs()
    demo()

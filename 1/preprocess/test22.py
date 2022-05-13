import cv2
import os
import skimage.io


def cropping(impath, destPath):
    # Read the input image
    img = cv2.imread(impath)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (0, 0, 255), 2)
        faces = img[y+10:y + h-10, x+10:x + w-10]
        #cv2.imshow("face",faces)
        resizedFace = cv2.resize(faces, (150,150))
        cv2.imwrite(destPath, resizedFace)


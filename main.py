import glob 
import os
import cv2
import sys
root_lib = '../../../../MyLibrary_py'
if root_lib not in sys.path:
    sys.path.append(root_lib)
import efdv_dip as dip
import math as mt
from keras.models import load_model
import numpy as np


def prepareImg(I):
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    _,thresholding  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    neg = dip.negative(thresholding)

    return gray, thresholding, neg

def segmentation(I):
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    u,v,c = I.shape 
    mthresholding = dip.multithresholding(gray, 7)
    mthresholding = mthresholding.astype('uint8')
    bw = np.zeros((u,v))
    for i in range(u):
        for j in range(v):
            if mthresholding[i,j] == 0:
                bw[i,j] = 255
    return bw

model = load_model('../results/results_ext3_m3_v1/model_CNN__3.h5')
root = '../../../datasets/TCIA_SegPC_dataset'
file = '/validation/validation/'
specifyfile = ['x/', 'y/']

dirfiles = glob.glob(root+file+specifyfile[0]+'/*')
names_files =  [os.path.basename(file_path) for file_path in dirfiles]
dirfiles_ref = glob.glob(root+file+specifyfile[1]+'/*')
names_files_ref =  [os.path.basename(file_path) for file_path in dirfiles_ref]

for names_imgs in names_files: 
    I = cv2.imread(root+file+specifyfile[0]+names_imgs)
    bw = segmentation(I)
    erode = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=6)
    bw, upcenters, downcenters, area = dip.centersbycompconn(erode, 5000)
    dilate = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=6)
    bw2, upcenters, downcenters, area = dip.centersbycompconn(dilate, 5000)
    for m in range(len(upcenters)):
        x,y = upcenters[m][0], upcenters[m][1]
        tamWindow = round(mt.sqrt(area[m]/mt.pi))
        crop = dip.cropbyfor(I, int(y),int(x), tamWindow+100, tamWindow+100)
        crop = cv2.resize(crop, (40,40))
        crop = crop/ 255.0
        crop = np.expand_dims(crop, axis=0)
        prediction = model.predict(crop)
        if prediction[0][0] > 0.6:
            x1, y1, x2, y2 = dip.getsidesquare((x,y), tamWindow+100, tamWindow+100)
            cv2.rectangle(I, (x1, y1), (x2, y2), (0, 255, 191), 2 )

        
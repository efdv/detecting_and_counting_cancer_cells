import sys
root_lib = '../../../../MyLibrary_py'
if root_lib not in sys.path:
    sys.path.append(root_lib)

import efdv_dip as dip
import cv2
import glob
import os
from skimage import measure
import numpy as np
import re
import math as mt

def prepareImg(I):
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    _,thresholding  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    neg = dip.negative(thresholding)

    return gray, thresholding, neg


root = '../../../datasets/TCIA_SegPC_dataset'
file = '/train/'
specifyfile = ['x/', 'y/']
dirfiles = glob.glob(root+file+specifyfile[0]+'/*')
names_files =  [os.path.basename(file_path) for file_path in dirfiles]
dirfiles_ref = glob.glob(root+file+specifyfile[1]+'/*')
names_files_ref =  [os.path.basename(file_path) for file_path in dirfiles_ref]


cont = 0
tamWindow = 350
for names_imgs in names_files: 
    print(names_imgs)
    words = re.split(r'[._]', names_imgs)

    I = cv2.imread(root+file+specifyfile[0]+names_imgs)
    gray,_,neg = prepareImg(I)


    imerode = cv2.erode(neg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=6)
    holes = dip.holes(imerode)

    bw, upcenters, downcenters = dip.centersbycompconn(holes, 20000) #este número se debe justificar
    x,y = upcenters[0][0], upcenters[0][1]


    for pathref in names_files_ref: 
        words_ref = re.split(r'[._]', pathref)
        if words_ref[0] == words[0]:
            print(pathref)
            Iref = cv2.imread(root+file+specifyfile[1]+pathref)
            bwref,_,_ = prepareImg(Iref)
            _,upcentersref,_ = dip.centersbycompconn(bwref)
            xr, yr = upcentersref[0][0], upcentersref[0][1]
            listofcenters = []
            for center in upcenters:
                x, y = center[0], center[1]
                d = mt.sqrt((xr-x)**2 + (yr-y)**2)
                if d > 1500:
                    left, top, right, bottom = dip.getsidesquare((x,y), tamWindow, tamWindow)
                    crop = dip.getCrops(I, left, top, right, bottom)
                    nameSave = str(cont) + '.png'
                    if (x,y) not in  listofcenters:
                        cv2.imwrite(root+'/crops/nocells/'+nameSave, crop)
                        listofcenters.append((x,y))
                    cont += 1




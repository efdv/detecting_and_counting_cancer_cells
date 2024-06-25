import cv2
import sys
root_lib = '../../../../MyLibrary_py'
if root_lib not in sys.path:
    sys.path.append(root_lib)
import efdv_dip as dip
import numpy as np

root = '../../../datasets/TCIA_SegPC_dataset/validation/validation/x/'
name = '9442.bmp'

I = cv2.imread(root+name)
u,v,c = I.shape 
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

mthresholding = dip.multithresholding(gray, 7)

mthresholding = mthresholding.astype('uint8')
bw = np.zeros((u,v))
for i in range(u):
    for j in range(v):
        if mthresholding[i,j] == 0:
            bw[i,j] = 255

#erode = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=6)

dip.showImgbyplt(bw)
dip.showImgbyplt(I)
# dip.showImgbyplt(thresholding_g)
# dip.showImgbyplt(dilate)




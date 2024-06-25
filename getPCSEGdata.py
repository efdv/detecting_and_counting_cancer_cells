import glob 
import cv2
from skimage import measure
import numpy as np
import re
import os


def showImg(img):
    cv2.imshow('imagen',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bycompconn(bw, tam):
    labeled_image = measure.label(bw, connectivity=2)
    regions = measure.regionprops(labeled_image)
    numPixels = [region.area for region in regions]
    #numPixels_sorted = sorted(numPixels, reverse=True)

    valpix = 255
    u,v = bw.shape
    bw2 = np.zeros((u,v))
    fcentros = []
    bcentros = []
    for i, region in enumerate(regions):
        if numPixels[i] > tam:
            centro_y, centro_x = region.centroid
            fcentros.append((centro_x, centro_y))
            bw2[region.coords[:, 0], region.coords[:, 1]] = valpix
        else: 
            centro_y, centro_x = region.centroid
            bcentros.append((centro_x, centro_y))


    bw_uint8 =  bw2.astype(np.uint8) * 255

    return bw_uint8, bcentros, fcentros


def getsquare(centro, w, h):
    x = centro[0][0]
    y = centro[0][1]
    left = int(x - w)
    top = int(y - h)
    right = int(x + w)
    bottom = int(y + h)
    return left, top, right, bottom

def getCrops(I, Iref, w,h):
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    _,thresholding = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    labeled_image = measure.label(thresholding, connectivity=2)
    region = measure.regionprops(labeled_image)
    centroid = [region.centroid for region in region]
    
    
    left, top, right, bottom = getsquare(centroid, w, h)
    if left <  0:
        left = 0
    elif top <  0:
        top = 0
    elif right < 0:
        right = 0
    elif bottom < 0:
        bottom = 0

    crop = Iref[left:right, top:bottom]

    return crop


root = '../../../datasets/TCIA_SegPC_dataset'
# root = r'C:\Users\ef.duquevazquez\OneDrive - Universidad de Guanajuato\EF-Duque-Vazquez-Doctorado\datasets\TCIA_SegPC_dataset'
file = '/validation/validation/'
specifyfile = ['x', 'y']


dirfiles_y = glob.glob(root+file+specifyfile[1]+'/*')
names_files =  [os.path.basename(file_path) for file_path in dirfiles_y]


cont = 0
for img in names_files:
    print(cont)
    I = cv2.imread(root+file+specifyfile[1]+'/'+img)
    words = re.split(r'[._]', img)
    name = words[0]+'.'+words[2]
    Iref = cv2.imread(root+file+specifyfile[0]+'/'+name)    
    crop = getCrops(I, Iref, 120, 120)
    nameSave = str(cont) + '.png'
    cv2.imwrite(root+'/crops/validation/'+nameSave, crop)
    cont += 1


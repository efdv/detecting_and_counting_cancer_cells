import cv2 
from skimage import measure
import numpy as np
import glob


def showImg(img):
    cv2.imshow('imagen',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def negative(bw):
    u,v = bw.shape
    for i in range(u):
        for j in range(v):
            if bw[i,j] == 0:
                bw[i,j] = 255
            else:
                bw[i,j] = 0
    return bw

def bycompconn(bw, tam):
    labeled_image = measure.label(thresholding_neg, connectivity=2)
    regions = measure.regionprops(labeled_image)
    numPixels = [region.area for region in regions]
    #numPixels_sorted = sorted(numPixels, reverse=True)

    valpix = 255

    bw = np.zeros((u,v))
    fcentros = []
    bcentros = []
    for i, region in enumerate(regions):
        if numPixels[i] > tam:
            centro_y, centro_x = region.centroid
            fcentros.append((centro_x, centro_y))
            bw[region.coords[:, 0], region.coords[:, 1]] = valpix
        else: 
            centro_y, centro_x = region.centroid
            bcentros.append((centro_x, centro_y))


    bw_uint8 =  bw.astype(np.uint8) * 255

    return bw_uint8, bcentros, fcentros

def saveimages(I, bcentros, fcentros, tamW):
    bcont = fcont = 0
    for bc, fc in zip(bcentros, fcentros):
        if bcont >=0 and bcont < 10 or fcont >=0 and fcont < 10 :
            pre = '00'
        elif bcont >= 10 and bcont < 100 or bcont >= 10 and bcont < 100:
            pre = '0'
        else:
            pre = ''
        # print("%i\t%i\t%i\t%i\t\n"%(int(bc[0]) - tamW, int(bc[0])+tamW, int(bc[1]) - tamW, int(bc[1])+tamW))
        # if ((int(bc[0]) - tamW) > 0 and (int(bc[0])+tamW) < u) and ((int(bc[1]) - tamW) > 0 and (int(bc[1])+tamW) < v) and ((int(fc[0]) - tamW) > 0 and (int(fc[0])+tamW) < u) and ((int(fc[1]) - tamW) > 0 and (int(fc[1])+tamW) < v): 
        #     broi = I[int(bc[0]) - tamW :int(bc[0])+tamW, int(bc[1])-tamW:int(bc[1])+tamW]
        #     froi = I[int(fc[0]) - tamW :int(fc[0])+tamW, int(fc[1])-tamW:int(fc[1])+tamW]
        # # froi = I[int(fc[1]):int(fc[1])+tamW, int(fc[0]):int(fc[0])+tamW]
        #     # u,v,c = broi.shape
        #     # if u == v:
        #     cv2.imwrite(rootSave+'impurities/'+pre+str(bcont)+'.jpg', broi)    
        #     # u,v,c = froi.shape
        #     # if u == v:
        #     cv2.imwrite(rootSave+'cells/'+pre+str(fcont)+'.jpg', froi)
            # bcont += 1
            # fcont += 1
        froi = I[int(fc[1]):int(fc[1])+tamW, int(fc[0]):int(fc[0])+tamW]
        broi = I[int(bc[1]):int(bc[1])+tamW, int(bc[0]):int(bc[0])+tamW]
        cv2.imwrite(rootSave+'cells/'+pre+str(fcont)+'.jpg', froi)   
        cv2.imwrite(rootSave+'impurities/'+pre+str(bcont)+'.jpg', broi)

        bcont += 1
        fcont += 1
        
        

def over( I,img_conn):
    img_over = np.zeros((u,v,c))
    img_over[:,:,0] = img_conn * I[:,:,0]
    img_over[:,:,1] = img_conn * I[:,:,1]
    img_over[:,:,2] = img_conn * I[:,:,2]

    return img_over

def byContours(contornos, tam):
    bcentros = []
    fcentros = []
    areas = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        areas.append(area)
        momentos = cv2.moments(contorno)
        if momentos["m00"] != 0:
            centro_x = int(momentos["m10"] / momentos["m00"])
            centro_y = int(momentos["m01"] / momentos["m00"])
            if area > tam:
                fcentros.append((centro_x, centro_y))
            else:
                bcentros.append((centro_x, centro_y))
    
    return bcentros, fcentros

def getsquare(centro, w, h):
    x, y = centro
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return x1, y1, x2, y2

#----------------------------------------------------------------------------------------------------------------------------------    
general_root = '../dataset/datasets/'
I = cv2.imread(general_root+'/Dataset_3/3-18.jpg')
gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
_,thresholding  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresholding_neg =  negative(thresholding)

kernel = np.ones((7, 7), np.uint8)
imerode = cv2.erode(thresholding_neg, kernel, iterations=2)

contornos, _ = cv2.findContours(imerode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bcentros, fcentros = byContours(contornos, 50)

img_conn, bcentros, fcentros = bycompconn(imerode, 30)


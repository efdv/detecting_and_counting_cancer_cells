import cv2
import glob

root = '../dataset/datasets/'
rootSave = '../dataset/full-dataset/' 
dirfile = glob.glob(root+'/*')
name = rootSave
cont = 0
for file in dirfile:
    dir_images = glob.glob(file+'/*')
    for path in dir_images:
        if cont >=0 and cont < 10:
            pre = '00'
        elif cont >= 10 and cont < 100:
            pre = '0'
        else:
            pre = ''    
        img = cv2.imread(path)
        cv2.imwrite(rootSave+pre+str(cont)+'.jpg', img)
        cont += 1
        print('Porcentaje de avance', (cont*100)/566)

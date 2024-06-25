import cv2
import glob


path = '../dataset/datasets/'
dirfile = glob.glob(path+'/*')

imgs = []
for ipath in dirfile:
    dir_images = glob.glob(ipath+'/*')
    for img in dir_images:
        I = cv2.imread(img)
        resized_image = cv2.resize(I, (224,224))
        imgs.append(resized_image)


cv2.imshow('test',imgs[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
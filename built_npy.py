import glob
import cv2
import numpy as np
import os
import re



# files = ['train', 'nocells' ]
# root = '../../../datasets/TCIA_SegPC_dataset/crops/'


# dataset = []
# labels = []
# cont = 0
# for f in files:
#     dirfile =  glob.glob(root+f+'/*')
#     for pathimg in dirfile:
#         I = cv2.imread(pathimg)
#         I = cv2.resize(I, (40, 40))
#         dataset.append(I)
#         labels.append(cont)
#     print(cont)
#     cont += 1
#     if cont > 1: cont = 0



# np.save(root+'dataset.npy', dataset)
# np.save(root+'labels.npy', labels)


#images for unet

root = '../../../datasets/TCIA_SegPC_dataset/validation'
specify_file = ['/x', '/y']
dirfiles = glob.glob(root+specify_file[0]+'/*')
names_files =  {os.path.basename(file_path): file_path for file_path in dirfiles}
dirfiles_gt = glob.glob(root+specify_file[1]+'/*')
names_files_gt =  {os.path.basename(file_path) :file_path  for file_path in dirfiles_gt}

listgts = []
listOr = []
total_files = len(names_files)
m = 1152
n = 1536
for cont, (names_imgs, img_path) in enumerate(names_files.items()): 
    
    words = names_imgs.split('.')[0]
    I = cv2.imread(img_path)
    I = cv2.resize(I, (m, n))
    u,v,c = I.shape
    gt = np.zeros((u,v), dtype='uint8')
    
    corresponding_gts = [name for name in names_files_gt if name.startswith(words)]
      
    for  gts in corresponding_gts:
        img_gt_path = names_files_gt[gts]
        img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.resize(img_gt, (m, n))
        _,thresholding  = cv2.threshold(img_gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gt += thresholding
    
    gt = cv2.resize(gt, (128, 128))
    I = cv2.resize(I, (128, 128))
    listgts.append(gt)
    listOr.append(I)


    print(f"{(cont / total_files) * 100:.2f}%")

listgts = np.array(listgts)
listOr = np.array(listOr)
np.save(root+'dataset.npy', listgts)
np.save(root+'labels.npy', listOr)


data = np.load("../../../datasets/TCIA_SegPC_dataset/crops/dataset.npy")
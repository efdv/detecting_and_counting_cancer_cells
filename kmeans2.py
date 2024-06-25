from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np 
import warnings
import glob
from sklearn.cluster import KMeans

path = '../dataset/datasets/'
dirfile = glob.glob(path+'/*')
dir_images = glob.glob(dirfile[0]+'/*')


image = imread(dir_images[0])

image.shape

plt.imshow(image)

x = image.reshape(-1,4)

kmeans = KMeans(n_clusters=2).fit(x)

segmented_image = kmeans.cluster_centers_[kmeans.labels_]

kmeans.cluster_centers_
kmeans.labels_

segmented_image = segmented_image.reshape(image.shape)

segmented_image = segmented_image.astype(np.uint8)
plt.imshow(segmented_image)


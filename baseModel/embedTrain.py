# -*- coding: utf-8 -*-
"""
Model definition:
    Global extractor created through transfer learning.
    Model is trained by using small amount of data.

Jon Wang, 2018 June 15.
"""


import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import Adam, SGD
from gdal import Open
from skimage.io import imread
from skimage import measure
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from imgaug import augmenters as iaa
import embedding
import sklearn.feature_extraction.image as im
import cv2


# ========== INPUT ==========
# Define input image dimensions
img_rows, img_cols = 32, 32
# The images are RGB.
img_channels = 3

# Use GDAL OPEN to read base RGB WorldView(WV02) map and corresponding reference label
wv = Open('mapMul.tif')
land = np.zeros((wv.RasterYSize,wv.RasterXSize,3)).astype('uint16')  #np.zeros_like(wv).astype('int16')
land[:,:,0] = wv.GetRasterBand(5).ReadAsArray()  # swaping the band and re-arrange the band order into 1-R, 2-G and 3-B
land[:,:,1] = wv.GetRasterBand(3).ReadAsArray()
land[:,:,2] = wv.GetRasterBand(2).ReadAsArray()
del wv  # delete variables to save memory resources
label = imread("labelMul.tif")

wv2 = Open('mapMul2.tif')
land2 = np.zeros((wv2.RasterYSize,wv2.RasterXSize,3)).astype('uint16')  #np.zeros_like(wv).astype('int16')
land2[:,:,0] = wv2.GetRasterBand(5).ReadAsArray()  # swaping the band and re-arrange the band order into 1-R, 2-G and 3-B
land2[:,:,1] = wv2.GetRasterBand(3).ReadAsArray()
land2[:,:,2] = wv2.GetRasterBand(2).ReadAsArray()
del wv2  # delete variables to save memory resources

# Convert to 0-255 valued UINT8 image
land = cv2.convertScaleAbs(land,alpha = 255.0/(np.max(land)))
land2 = cv2.convertScaleAbs(land2,alpha = 255.0/(np.max(land2)))
slum = (label > 0).astype('uint8')
slum2 = np.zeros((land2.shape[0],land2.shape[1])).astype('uint8')
del label  # delete variables to save memory resources
#plt.imshow(slum, 'gray')
land_slum = np.dstack((land, slum))  # stack the map and label for drawing training patches
land_slum2 = np.dstack((land2, slum2))

# Sample from slum blocks according to the size of connected slum blocks
slum_block = measure.label(slum, background = False, connectivity = 1)  # extract and label connected slum blocks
regions = measure.regionprops(slum_block)  # get the properties of labelled slum blocks
slum_areas = []  # store the size of area of labelled slum blocks for sorting and choosing large slum blocks as samples

for region in regions:  # query the slum block area property and store the sizes
    slum_areas.append(region.area)
sort_index = [b[0] for b in sorted(enumerate(slum_areas),key=lambda i:i[1])]  # sort the sizes of slum blocks

sample_block = 4  # draw samples from several numbers of slum blocks
# slum_patches = 1000  # number of training patches drawn from sample slum blocks 
non_slum_patches = 2000
np.random.get_state()  # create pseudo random state to draw training patches from the bounding sample area
rng = np.random.RandomState(0)

for j in range (sample_block):
    slum_bounding = regions[sort_index[np.shape(slum_areas)[0]-(j+1)]].bbox  # identify the bounding of the largest slum block
    # plt.imshow(land_slum[slum_bounding[0]:slum_bounding[2], slum_bounding[1]:slum_bounding[3], 3])
    # draw randomly sample patches for training and testing
    slum_patches = int((slum_bounding[2]-slum_bounding[0])*(slum_bounding[3]-slum_bounding[1])/5)  # number of training patches drawn from sample slum blocks 
    if j < 1:
        datasets0 = im.extract_patches_2d(land_slum[slum_bounding[0]:slum_bounding[2], slum_bounding[1]:slum_bounding[3], :],(img_rows,img_cols),slum_patches, random_state = rng)
    else:
        datasetsTemp = im.extract_patches_2d(land_slum[slum_bounding[0]:slum_bounding[2], slum_bounding[1]:slum_bounding[3], :],(img_rows,img_cols),slum_patches, random_state = rng)
        datasets0 = np.concatenate((datasets0, datasetsTemp), axis=0)      
datasets1 = im.extract_patches_2d(land_slum[1700:2000, 1990:2200, :],(img_rows,img_cols),non_slum_patches, random_state = rng)
datasets2 = im.extract_patches_2d(land_slum[1600:1850, 3400:3600, :],(img_rows,img_cols),non_slum_patches, random_state = rng)
datasets3 = im.extract_patches_2d(land_slum2[1600:1900, 1000:1300, :],(img_rows,img_cols),non_slum_patches, random_state = rng)
datasets = np.concatenate((datasets0, datasets1, datasets2, datasets3),axis=0)
np.random.shuffle(datasets)  # randomly shuffle the patches
del land_slum, land, slum, land_slum2, land2, slum2, slum_block, datasetsTemp, datasets0, datasets1, datasets2, datasets3  # delete variables to save memory resources

# Image Augmentation !!
# Sometimes(0.8, ...) applies the given augmenter in 80% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.6, aug)

seq = iaa.Sequential([
    sometimes(iaa.Crop(px=(0, 3))), # crop images from each side by 0 to 16px (randomly chosen)
    sometimes(iaa.Fliplr(1)), # horizontally flip 50% of the image
    sometimes(iaa.GaussianBlur(sigma=(0, 0.1))), # blur images with a sigma of 0 to .1
    sometimes(iaa.Affine(
                         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                         #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                         rotate=(-45, 45), # rotate by -45 to +45 degrees
                         shear=(-10, 10), # shear by -10 to +10 degrees
                         #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
              )),
    sometimes(iaa.PiecewiseAffine(scale=(0, 0.03))),
    sometimes(iaa.PerspectiveTransform(scale=(0, 0.1)))
])

datasets = seq.augment_images(datasets)
datasets = datasets.astype('float32')
np.random.shuffle(datasets)


# ========== COMPILE ==========
# dimensions of our images.
nb_classes = 256

embed = embedding.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

label = datasets[:,:,:,img_channels]
datasets = datasets[:,:,:,0:img_channels]
mean_image = np.mean(datasets, axis=0)
datasets -= mean_image
datasets /= 128.

def coverage(img):
    """Compute the coverage of the target pixels over a whole image as ratio.
    If the coverage ratio exceeds the customized threshold, the image should be
    labeled as a positive target with value '1', otherwise '0'.
    """
    return (img > 0).sum()/(img.shape[0]*img.shape[1])

labelNew = np.zeros((label.shape[0])).astype('float32')
for i in range(label.shape[0]):
    labelNew[i] = coverage(label[i,:,:])
del label    

# Encode label into binary categoricals
# Spans through 0~255 but compressed to 250~255
spread = 3  # spread the 0~1 to 0~spread interval and rise it to 255
labelNew = (labelNew*spread+(labelNew>(1.0/spread))*(255-spread)).astype('int')
labelNew = to_categorical(labelNew)

X_train = datasets[:int(datasets.shape[0]*0.7),:,:,:]
Y_train = labelNew[:int(labelNew.shape[0]*0.7)]
X_val = datasets[int(datasets.shape[0]*0.7):datasets.shape[0],:,:,:]
Y_val = labelNew[int(labelNew.shape[0]*0.7):labelNew.shape[0]]
del datasets
del labelNew
    

# ========== TRAIN ==========
# Design top model
batch_size = 32
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('EmbedLog4.csv')

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
embed.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
metrics=['accuracy'])
# Save the embed model architecture as Embed
embed_json = embed.to_json()
with open("Embed.json","w") as json_file:
    json_file.write(embed_json)

embed.fit(X_train, Y_train,
          epochs=30,
          batch_size=batch_size,
          validation_data=(X_val, Y_val))

embed.save_weights('EmbedWeights.h5')


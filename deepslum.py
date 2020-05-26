#-------------------------------------------------------------------------------
# Name:        Deep Convolutional Neural Network (DCNN) for slum mapping
# Purpose:     Test DCNN on mapping degrees of deprivation
#              Main method is ...

# Version:     0.1 
#              Functionalities:
#              1.
#              2.
#              3.
#              4.

# Author:      Jiong (Jon) Wang
#
# Created:     26/05/2020
# Copyright:   (c) JonWang 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------


from keras import backend as K
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Input,
    Conv2DTranspose,
    Flatten,
    BatchNormalization,
    Activation,
    concatenate
)
from keras.callbacks import TensorBoard 
from keras.layers.core import RepeatVector, Reshape
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage import measure
from skimage.util.shape import view_as_blocks
from imgaug import augmenters as iaa
from scipy import interpolate
from gdal import Open, GA_Update
from osgeo import gdal_array
from pathlib import Path
from PIL import Image
import sklearn.feature_extraction.image as im
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tensorflow as tf


##############################################
# 01 Load and organize images
##############################################

# Stack base image and label into a list of array
def load_image_label(path, subUL, extent):
    """Load training data pairs (two high resolution images and two low resolution images)"""
    path_list = []  # List image name and path
    for path in Path(path).glob('*.tif'):
        path_list.append(path)
    assert len(path_list) == 2

#    for path in path_list:  # Organize path list
#        img_name = path.name
#        if 'base' in img_name:
#            base = path
#        elif 'label' in img_name:
#            label = path
#    path_list = [base, label]
    
    stack = []  # Stack base and label together into a 3D array
    for path in path_list:
        if path.name.startswith('base'):
            data = gdal_array.LoadFile(str(path), xoff=subUL[0], yoff=subUL[1], 
                                       xsize=extent, ysize=extent) #.astype(np.int)
            for band in range(data.size[2]):
                stack.append(band)
        else:
            data = gdal_array.LoadFile(str(path), xoff=subUL[0], yoff=subUL[1], 
                                       xsize=extent, ysize=extent) #.astype(np.int)
#        image = Image.fromarray(data)
#        data = nan_remover(data)
#        setattr(image, 'filename', path)
        stack.append(data)
    # Ensure the size of base and label is are consistent
    assert stack[0].size == stack[-1].size
    return stack[:stack.size[2]-1], stack[:-1]


# Clean the NaN values
def nan_remover(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # Masking invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # Getting only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    array_interp = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy), method='nearest')
    # Clean the edge
    bad_indexes = np.isnan(array_interp)
    good_indexes = np.logical_not(bad_indexes)
    good_data = array_interp[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    array_interp[bad_indexes] = interpolated
    return array_interp


##############################################
# 02 Prepare training data
##############################################

# Image patches for training the model
# Stride controls the overlap of patches
def gen_patches(image, size, stride=None):
    """Segment input images into patches"""
    if not isinstance(size, tuple):  # Ensure format is tuple
        size = (size, size)
    if stride is None:
        stride = size
    elif not isinstance(stride, tuple):
        stride = (stride, stride)
    # Columns in priority
    for i in range(0, image.shape[0] - size[0] + 1, stride[0]):  # One patch every stride
        for j in range(0, image.shape[1] - size[1] + 1, stride[1]):
            yield image[i:i + size[0], j:j + size[1], np.newaxis]  # If Pillow Image is used: image.crop([i, j, i + size[0], j + size[1]])


# 
def sample_to_array(stack, lr_gen_sub, hr_gen_sub, patches):
    # samples are image pairs to be cutted into patches and fed into the CNN
#    assert len(stack) == 4
    for i in range(4):
        if i % 2 == 0:
            patches[i] += [img for img in lr_gen_sub(samples[i])]  # If Pillow Image: img_to_array(img)
        else:
            patches[i] += [img for img in hr_gen_sub(samples[i])]


def load_train_set(data_dir, scale, lr_sub_size, lr_sub_stride=2, 
                   pred_suffix='pred', valid_suffix='valid', input_suffix='input', modis_prefix='MOD11A1', landsat_prefix='LC08'):
    """Load image data from training folder (obtain low resolution data by sampling from high resolution data)"""
    hr_sub_size = lr_sub_size * scale
    hr_sub_stride = lr_sub_stride * scale

    lr_gen_sub = partial(gen_patches, size=lr_sub_size, stride=lr_sub_stride)
    hr_gen_sub = partial(gen_patches, size=hr_sub_size, stride=hr_sub_stride)

    patches = [[] for _ in range(4)]
    for path in (data_dir/'Train').glob('*'):
        if path.is_dir():
            print('loading image pairs from {}'.format(path))
            samples = load_image_pairs(path, pred_suffix, valid_suffix, input_suffix, modis_prefix, landsat_prefix, scale=scale)
            samples = [*samples[0], samples[1]]
            # subset samples into patches
            sample_to_array(samples, lr_gen_sub, hr_gen_sub, patches)
            
    # Split patches into training and validation sets        
    patch_train = [[] for _ in range(4)]
    patch_val = [[] for _ in range(4)]
    for i in range(4):
        patch_train[i] = np.stack(patches[i][:int(len(patches[i])*0.7)])
        patch_val[i] = np.stack(patches[i][int(len(patches[i])*0.7):])
    # Return 4-dimensional array (number, height, width, channel)
    return patch_train[:3], patch_train[-1], patch_val[:3], patch_val[-1]







##############################################
# Main
##############################################

def main():
    
    # Input training patch dimensions
    img_rows, img_cols = 32, 32
    # The images are 3-channel (RGB).
    img_channels = 3
    
    # Subset study area
    subUL = [, ]
    extent = 






##############################################
# Execution
##############################################
if __name__ == '__main__':
    main()
# Define input image dimensions
img_rows, img_cols = 32, 32
# The images are RGB.
img_channels = 3

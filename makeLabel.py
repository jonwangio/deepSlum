# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:23:04 2020

@author: Jon
"""
from osgeo import gdal
import numpy as np

image_path = 'sample_data2/test/image/0.tif'
#Open our original data as read only
dataset = gdal.Open(image_path, gdal.GA_ReadOnly)

#Note that unlike the rest of Python, Raster Bands in GDAL are numbered
#beginning with 1.
#I suspect this is to conform to the landsat band naming convention
band = dataset.GetRasterBand(1)

#Read in the data from the band to a numpy array
data = band.ReadAsArray()
data = data.astype(np.float)

#Use numpy, scipy, and whatever Python to make some output data
#That for ease of use should be an array of the same size and dimensions
#as the input data.
out_data = np.where(abs(data) > 0, 0, 0)
#Note -9999 is a convenience value for null - there's no number for
#transparent values - it's just how you visualise the data in the viewer

#And now we start preparing our output
driver = gdal.GetDriverByName("GTiff")
metadata = driver.GetMetadata()

#Create an output raster the same size as the input
out = driver.Create("out_file.tif",
                    dataset.RasterXSize,
                    dataset.RasterYSize,
                    1, #Number of bands to create in the output
                    gdal.GDT_Float32)

#Copy across projection and transform details for the output
out.SetProjection(dataset.GetProjectionRef())
out.SetGeoTransform(dataset.GetGeoTransform()) 

#Get the band to write to
out_band = out.GetRasterBand(1)

#And write our processed data
out_band.WriteArray(out_data)


##############################################
# Patch to boundary
##############################################

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extractEdges(im):
    # Otsu's method
    high_thresh, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    edges = cv2.Canny(im,low_thresh,high_thresh,apertureSize=3)  # Canny only accepts int
    print(edges.shape)
    plt.imshow(edges,'gray')
    return edges

def grow(im):
    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((3,3), np.uint8)
    im_dilation = cv2.dilate(im, kernel, iterations=1)
    plt.imshow(im_dilation,'gray')
    return im_dilation


path = Path('./label/')
for file in Path(path).glob('*.tif'):
    im = cv2.imread(str(file),0)
    im_edge = extractEdges(im)
    im_dilated = (grow(im_edge)/255).astype(int)
    
    im_non = (im + im_dilated < 1).astype(int)
    
    im_true = (im_dilated + im_non <1).astype(int)
    
#    fig = plt.figure(figsize=(12, 6))
#    plt.subplot(1,2,1), plt.imshow(im_true)
#    plt.subplot(1,2,2), plt.imshow(im_non)
    
    im_stack = np.dstack((im_true, im_dilated, im_non))
    
    cv2.imwrite(file.name, im_stack)








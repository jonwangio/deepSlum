# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:53:35 2020

@author: Jon
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################
# 00 Thresholding, edges, lines and shapes
##############################################

# Edges from gradient
def extractEdges(im):
    # Otsu's method
    high_thresh, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    edges = cv2.Canny(im,low_thresh,high_thresh,apertureSize=3)  # Canny only accepts int
    cv2.imshow("Window", edges)
    return edges
    
# Lines from edges
def extractLines(edges, im):
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,np.array([]),
                            minLineLength,maxLineGap)
    print('Number of lines detected: ', len(lines))
    
    # Visualization
    # Bacground image to draw lines
    backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    # Draw lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(backtorgb, (x1, y1), (x2, y2), (255,0,0), 1)            
    plt.imshow(backtorgb)
    return lines


##############################################
# 01 Contours
##############################################
    
# Contours from edges
def extractCnt(im):
    kernel = np.ones((5,5))
    imm = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    # find contours
    ret,thresh = cv2.threshold(imm,200,255,cv2.THRESH_BINARY_INV)
    cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
#    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Visualization
    # Bacground image to draw lines
    # Visualization
    backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)

    for cnt in cnts:
        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 40000:
            cv2.drawContours(backtorgb,[cnt],0,(127),-1)
    
    plt.imshow(backtorgb)
    return cnts


##############################################
# 02 Bounding methods: box, polygons
##############################################

# Minimum bounding rectangle
def minRec(cnts, im):
    # Visualization
    # Bacground image to draw lines
    # Visualization
    backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)

    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(backtorgb, [box], 0, (255,0,0), 2) # this was mostly for debugging you may omit
        plt.show()
        
    plt.imshow(backtorgb)
    return 

# Greedy Douglas-Peucker algorithm based from OpenCV
def polyDouglas(cnt):
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, epsilon, closed=False)
    cv2.imshow("Window", poly)
    return poly


##############################################
# 02 Run
##############################################

# Loading prediction
pred = np.load('pred.npy')
array = pred[0,:,:,0]  # extract 2D array from the prediction
im = np.array(array*255, dtype = np.uint8)  # convert to 8bit int
# Plot input
cv2.imshow("Window", im)
plt.imshow(im,'gray')

edges = extractEdges(im)
lines = extractLines(edges,im)
cnts = extractCnt(edges,im)
poly = polyDouglas(cnts)

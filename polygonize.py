# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:53:35 2020

@author: Jon
"""

import cv2
import numpy as np
import math
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

# Skeleton
def findSkel(im):
    skeleton = np.zeros(im.shape,np.uint8)
    eroded = np.zeros(im.shape,np.uint8)
    temp = np.zeros(im.shape,np.uint8)

    _,thresh = cv2.threshold(im,127,255,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
#            return (skeleton,iters)
            plt.imshow(skeleton,'gray')
            return skeleton

# Lines from edges
def extractLines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                            minLineLength=5, maxLineGap=10)
    print('Number of lines detected: ', len(lines))
    
    # Visualization
    # Bacground image to draw lines
    backtorgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    # Draw lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(backtorgb, (x1, y1), (x2, y2), (255,0,0), 1)            
    plt.imshow(backtorgb)
    return lines

# Line process bundler
class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def distance_to_line(self, point, line):
        """Get distance between point and line
        https://stackoverflow.com/questions/40970478/python-3-5-2-distance-from-a-point-to-a-line
        """
        px, py = point
        x1, y1, x2, y2 = line
        x_diff = x2 - x1
        y_diff = y2 - y1
        num = abs(y_diff * px - x_diff * py + x2 * y1 - y2 * x1)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den
    
    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.distance_to_line(a_line[:2], b_line)
        dist2 = self.distance_to_line(a_line[2:], b_line)
        dist3 = self.distance_to_line(b_line[:2], a_line)
        dist4 = self.distance_to_line(b_line[2:], a_line)
    
        return min(dist1, dist2, dist3, dist4)
    
    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 10
        min_angle_to_merge = 10
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, im):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation < 135:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
            if len(i) > 0:
                groups = self.merge_lines_pipeline_2(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_lines_segments1(group))
                merged_lines_all.extend(merged_lines)
        # Visualization
        backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        for line in merged_lines_all:
            cv2.line(backtorgb, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (255,0,0), 1)
        plt.imshow(backtorgb,'gray')

        return merged_lines_all
    
    
##############################################
# 01 Contours
##############################################
    
# Contours from edges
def extractCnt(im, thrsl, thrsh):
    kernel = np.ones((1,1))
    imm = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    # find contours
    ret,thresh = cv2.threshold(imm,thrsl,thrsh,cv2.THRESH_BINARY_INV)
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

# Contour features wanted
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2))

"""
def find_shapes(im):
#    im = cv2.GaussianBlur(im, (3,3), 0)
    kernel = np.ones((9,9))
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    shapes = []
    for thrs in range(15, 246, 5):
        print('Threshold at:', thrs)
        if thrs == 0:
            bina = cv2.Canny(im, 0, 50, apertureSize=5)
            bina = cv2.dilate(bina, None)
        else:
            _retval, bina = cv2.threshold(im, thrs, thrs+11, cv2.THRESH_BINARY)
#        contours, hier = cv2.findContours(bina, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours, hier = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.01*cnt_len, True)
            if 3<len(cnt)<30 and 500<cv2.contourArea(cnt)<30000:  # and cv2.isContourConvex(cnt):
                l = len(cnt)
                cnt = cnt.reshape(-1, 2)
#                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1)%l], cnt[(i+2)%l] ) for i in range(l)])
#                    if max_cos < 0.3:
#                        shapes.append(cnt)
                num = sum(angle < 0.1 for angle in [angle_cos( cnt[i], cnt[(i+1)%l], cnt[(i+2)%l] ) for i in range(l)])
                if num > 3:
                    shapes.append(cnt)

    backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(backtorgb, shapes, -1, (255, 0, 0), 1)
    plt.imshow(backtorgb)

    return shapes
"""

def find_shapes(im):
    shapes = []
#    bina = cv2.adaptiveThreshold(imm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv2.THRESH_BINARY,299,2)
    bina = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,99,-5)
    plt.imshow(bina, 'gray')
    cv2.imshow('Adaptive Binary', bina) 
    contours, hier = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.01*cnt_len, True)
        if 3<len(cnt)<200 and 50<cv2.contourArea(cnt)<30000:  # and cv2.isContourConvex(cnt):
            l = len(cnt)
            cnt = cnt.reshape(-1, 2)
#                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1)%l], cnt[(i+2)%l] ) for i in range(l)])
#                    if max_cos < 0.3:
#                        shapes.append(cnt)
            num = sum(angle < 0.1 for angle in [angle_cos( cnt[i], cnt[(i+1)%l], cnt[(i+2)%l] ) for i in range(l)])
            if num > 3:
                shapes.append(cnt)

    backtorgb = cv2.cvtColor(bina,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(backtorgb, shapes, -1, (255, 0, 0), 2)
    plt.imshow(backtorgb,'gray')
    cv2.imshow('Shapes', backtorgb)

    return shapes

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
        cv2.drawContours(backtorgb, [box], 0, (255,0,0), 1) # this was mostly for debugging you may omit
        plt.show()
        
    plt.imshow(backtorgb)
    return 

# Greedy Douglas-Peucker algorithm based from OpenCV
def polyDouglas(cnts, im):
    # Bacground image for visualization
    backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)

    for cnt in cnts:
        epsilon = 0.009 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        cv2.drawContours(backtorgb, [approx], -1, (255,0,0), 1)

    plt.imshow(backtorgb, 'gray')
    return approx


##############################################
# 02 Run
##############################################

# Loading prediction
pred = np.load('pred_na_small.npy')
array = pred  #[0,:,:,0]  # extract 2D array from the prediction
im = np.array(array*255, dtype = np.uint8)  # convert to 8bit int
# Plot input
plt.imshow(im,'gray')

edges = extractEdges(im)

skel = findSkel(im)

lines = extractLines(edges)

h = HoughBundler()
merged_lines = h.process_lines(lines, im)

t1, t2 = 90, 100
cnts = extractCnt(im, t1, t2)

poly = polyDouglas(cnts, im)

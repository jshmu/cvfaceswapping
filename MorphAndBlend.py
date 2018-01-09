#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:13:40 2017

@author: anant
"""
import numpy as np
import cv2
import scipy
from warpTriangle import warpTriangle

def MorphAndBlend(weighted_reshaped, shape_jane, frame, jane_image):

    beforeshape = weighted_reshaped
    shape1 = np.append(weighted_reshaped,np.array([[0,0],[0,frame.shape[0]-1],[frame.shape[1]-1,0], [frame.shape[1]-1,frame.shape[0]-1]]), axis=0)
    shape_jane1 = np.append(shape_jane,np.array([[0,0],[0,jane_image.shape[0]-1],[jane_image.shape[1]-1,0], [jane_image.shape[1]-1,jane_image.shape[0]-1]]), axis=0)

    Tri1 = scipy.spatial.Delaunay(shape1)
    Tri2 = scipy.spatial.Delaunay(shape_jane1)
    warped_image=np.copy(frame)
    for tri in Tri1.simplices:
        warpTriangle(jane_image,warped_image,shape_jane1[tri],shape1[tri])
    hullIndex = cv2.convexHull(beforeshape, returnPoints = False)
    hull2 =[]
    i=0
    for i in range(0, len(hullIndex)):
        hull2.append(beforeshape[hullIndex[i]][0])
    
    hullmask = []
    i=0
    
    hull2 = np.asarray(hull2)
    for i in range(len(hull2)):
        hullmask.append((hull2[i,0], hull2[i,1]))
    
    mask = np.ones(frame.shape, dtype = frame.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hullmask), (255, 255, 255))
    
    rect = cv2.boundingRect(np.float32([hull2]))    
    
    center = ((rect[0]+int(rect[2]/2), rect[1]+int(rect[3]/2)))
    
    # Clone seamlessly
    blended = cv2.seamlessClone(warped_image, frame, mask, center, cv2.NORMAL_CLONE)   
    
    return blended
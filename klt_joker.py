#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 02:34:40 2017

@author: anant
"""

import numpy as np
import cv2
import dlib
from imutils import face_utils
from MorphAndBlend import MorphAndBlend


frame_middle = cv2.imread('CIS581Project4PartCDatasets/Hard/frame-61.jpg')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

gray_middle = cv2.cvtColor(frame_middle, cv2.COLOR_BGR2GRAY)

rects_middle = detector(gray_middle, 1)
for (i, rect) in enumerate(rects_middle):
    shape_middle = predictor(gray_middle, rect)
    shape_middle = face_utils.shape_to_np(shape_middle)
    
p_old = shape_middle.reshape([68,1, 2])
    
p_old = p_old.astype(np.float32)
weighted_previous = p_old

points_backwards = np.zeros([61,68,1,2])

for i in range(60,0,-1):
    points_backwards[i] = p_old
    weighted_new = np.zeros([68,1,2])
    filename = 'CIS581Project4PartCDatasets/Hard/frame-' + str(i) + '.jpg'
    frame = cv2.imread(filename)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p_old = p_old.astype(np.float32)
    p_new, st, err = cv2.calcOpticalFlowPyrLK(gray_middle, frame_gray, p_old, None, **lk_params)
    rects_new = detector(frame_gray, 1)
    for (i, rect) in enumerate(rects_new):
        shape_new = predictor(frame_gray, rect)
        shape_new = face_utils.shape_to_np(shape_new)
        
        #making sure it works for one face; ignore any other faces detected
        if (i == 1):
            break
        
    face_pts_new = shape_new.reshape(68, 1, 2)
    
    dlib_flag = 0
#        print shape_new.shape[0]
#        print st.shape
    if shape_new.shape[0] == 68:
        #print "helllo"
        dlib_flag = 1
        
        
    for pt_num in range(68):
        if (dlib_flag == 1 and st[pt_num] == 1):
            #print "path1"
            weighted_new[pt_num, :, :] = 0.2*face_pts_new[pt_num, :, :] + 0.8*p_new[pt_num, :, :]
        elif (dlib_flag == 1 and st[pt_num] == 0):
            #print "path2"
            weighted_new[pt_num, :, :] = 0.8*weighted_previous[pt_num, :, :] + 0.2*face_pts_new[pt_num, :, :]
        elif (dlib_flag == 0 and st[pt_num] == 1):
            #print "path3"
            weighted_new[pt_num, :, :] = 0.2*weighted_previous[pt_num, :, :] + 0.8*p_new[pt_num, :, :]
        else:
            #print "path4"
            weighted_new[pt_num, :, :] = weighted_previous[pt_num, :, :]
    
    weighted_reshaped = weighted_new.reshape(68,2)        
    for (x, y) in weighted_reshaped:
        cv2.circle(frame, (np.int(x), np.int(y)), 1, (0, 0, 255), -1)
        
    cv2.imshow('frame_Check', frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    # Now update the previous frame and previous points
    gray_middle = frame_gray.copy()
    p_old = weighted_new.reshape(-1,1,2) 
    

# ----------------------------------------------------------------

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap2 = cv2.VideoCapture('Anant.mp4')
#cap2.set(1,5)

ret2, jane_image = cap2.read()
#jane_image = cv2.imread('image1.jpg')
gray = cv2.cvtColor(jane_image, cv2.COLOR_BGR2GRAY)
rects_jane = detector(gray, 1)
for (i, rect) in enumerate(rects_jane):
    shape_jane = predictor(gray, rect)
    shape_jane = face_utils.shape_to_np(shape_jane)

cap1 = cv2.VideoCapture('CIS581Project4PartCDatasets/Hard/Joker.mp4')
 
# Check if camera opened successfully
if (cap1.isOpened() == False): 
  print("Unable to read camera feed")
 
frame_width = int(cap1.get(3))
frame_height = int(cap1.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))


ret, old_frame = cap1.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#Contrast enhancement using clahe 
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#old_gray = clahe.apply(old_gray)


for i in range(60):
    weighted_new = np.zeros([68,1,2])
    ret, frame = cap1.read()

    if ret == 1:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        weighted_reshaped = points_backwards[i+1].reshape([68,2]).astype(int)
        fuckitshape = weighted_reshaped
        
        frame_check = np.copy(frame)
        
        for (x, y) in weighted_reshaped:
		cv2.circle(frame_check, (x, y), 1, (0, 0, 255), -1)
        
        blended = MorphAndBlend(weighted_reshaped, shape_jane, frame, jane_image)                

        out.write(blended)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


cap1.set(1,61)
while(1):
    weighted_new = np.zeros([68,1,2])
    ret, frame = cap1.read()

    if ret == 1:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        p_old = p_old.astype(np.float32)
        # calculate optical flow
        p_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p_old, None, **lk_params)
    
        rects_new = detector(frame_gray, 1)
        for (i, rect) in enumerate(rects_new):
            shape_new = predictor(frame_gray, rect)
            shape_new = face_utils.shape_to_np(shape_new)
            
            #making sure it works for one face; ignore any other faces detected
            if (i == 1):
                break
            
        face_pts_new = shape_new.reshape(68, 1, 2)
        
        dlib_flag = 0
        if shape_new.shape[0] == 68:
            #print "helllo"
            dlib_flag = 1
            
        for pt_num in range(68):
            if (dlib_flag == 1 and st[pt_num] == 1):
                #print "path1"
                weighted_new[pt_num, :, :] = 0.5*face_pts_new[pt_num, :, :] + 0.5*p_new[pt_num, :, :]
            elif (dlib_flag == 1 and st[pt_num] == 0):
                #print "path2"
                weighted_new[pt_num, :, :] = 0.5*weighted_previous[pt_num, :, :] + 0.5*face_pts_new[pt_num, :, :]
            elif (dlib_flag == 0 and st[pt_num] == 1):
                #print "path3"
                weighted_new[pt_num, :, :] = 0.5*weighted_previous[pt_num, :, :] + 0.5*p_new[pt_num, :, :]
            else:
                #print "path4"
                weighted_new[pt_num, :, :] = weighted_previous[pt_num, :, :]
                
        weighted_reshaped = weighted_new.reshape([68,2]).astype(int)
        fuckitshape = weighted_reshaped
        
        
        blended = MorphAndBlend(weighted_reshaped, shape_jane, frame, jane_image)                 
        
        if (len(st[st == 1]) < 45):
            out.write(frame)
        else:
            out.write(blended)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p_old = weighted_new.reshape(-1,1,2)
        
    else:
        break
cv2.destroyAllWindows()
cap1.release()
out.release()
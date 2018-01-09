import numpy as np
import cv2
import dlib
from imutils import face_utils
from MorphAndBlend import MorphAndBlend

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap2 = cv2.VideoCapture('Jane2.mp4')

ret2, jane_image = cap2.read()

gray = cv2.cvtColor(jane_image, cv2.COLOR_BGR2GRAY)
rects_jane = detector(gray, 1)
for (i, rect) in enumerate(rects_jane):
    shape_jane = predictor(gray, rect)
    shape_jane = face_utils.shape_to_np(shape_jane)

cap1 = cv2.VideoCapture('CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4')
 
# Check if camera opened successfully
if (cap1.isOpened() == False): 
  print("Unable to read camera feed")
 
frame_width = int(cap1.get(3))
frame_height = int(cap1.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap1.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#Contrast enhancement using clahe 
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#old_gray = clahe.apply(old_gray)

#Contrast Enhancement using Histogram Equalization
#old_gray = cv2.equalizeHist(old_gray)


rects_first = detector(old_gray, 1)
for (i, rect) in enumerate(rects_first):
    shape_first = predictor(old_gray, rect)
    shape_first = face_utils.shape_to_np(shape_first)
    #making sure it works for one face; ignore any other faces detected
    if(i == 1):
        break

face_pts = shape_first.reshape([68,1, 2])

p_old = face_pts.astype(np.float32)
weighted_previous = p_old

# Use facial landmarks as previous points for Optical FLow
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
#        print shape_new.shape[0]
#        print st.shape
        if shape_new.shape[0] == 68:
            #print "helllo"
            dlib_flag = 1
        
        #for each point, compute a runnin average with higher weight on KLT
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
                
        weighted_reshaped = weighted_new.reshape([68,2]).astype(int)
        beforeshape = weighted_reshaped
        weighted_reshaped_copy = np.copy(weighted_reshaped)
        shape_jane_copy = np.copy(shape_jane)
        frame_copy = np.copy(frame)
        jane_image_copy = np.copy(jane_image)
        
        #Warping using triangulation and affine transformation
        
        blended = MorphAndBlend(weighted_reshaped_copy, shape_jane_copy, frame_copy, jane_image_copy)             

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
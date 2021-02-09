import numpy as np
import cv2
from helper.utils import points2array

def overlay_heart_eyes(input_img,landmark_files):
  detector,predictor = landmark_files
  msg = None

  heart = cv2.imread("./helper/heart.jpg")
  detections = detector(input_img,1)
  if len(detections) == 0:
    msg = "Face not found. Try re-aligning your face."
    return input_img,msg

  for rect in detections:
    landmarks = predictor(input_img,rect)
    landmarks = points2array(landmarks.parts())
    le_upper_y = (landmarks[37][1] + landmarks[19][1]) //2
    le_lower_y = (landmarks[41][1] + landmarks[30][1])// 2
    le_left_x = (landmarks[17][0] + landmarks[36][0]) //2
    le_right_x = (landmarks[21][0] + landmarks[39][0]) //2

    re_upper_y = (landmarks[44][1] + landmarks[24][1]) //2
    re_lower_y = (landmarks[46][1] + landmarks[30][1])// 2
    re_left_x = (landmarks[26][0] + landmarks[45][0]) //2
    re_right_x = (landmarks[22][0] + landmarks[42][0]) //2

    l_ul = (le_left_x,le_upper_y)
    l_ur = (le_right_x,le_upper_y)
    l_lr = (le_right_x,le_lower_y)
    l_ll = (le_left_x,le_lower_y)

    r_ul = (re_left_x,re_upper_y)
    r_ur = (re_right_x,re_upper_y)
    r_lr = (re_right_x,re_lower_y)
    r_ll = (re_left_x,re_lower_y)

    pts_src = np.asarray(([(0,0),(heart.shape[1],0),(heart.shape[1],heart.shape[0]),(0,heart.shape[0])]))

    pts_dst = np.asarray([l_ul,l_ur,l_lr,l_ll])
    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]
    le_mask_out = cv2.warpPerspective(heart,H,(input_img.shape[1],input_img.shape[0]))

    pts_dst = np.asarray([r_ul,r_ur,r_lr,r_ll])
    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]
    re_mask_out = cv2.warpPerspective(heart,H,(input_img.shape[1],input_img.shape[0]))

    final_mask = le_mask_out + re_mask_out
    input_img_mask = cv2.inRange(final_mask,(0,0,0),(0,0,255))
    input_img = cv2.bitwise_and(input_img,input_img,mask = input_img_mask) + final_mask

  return input_img,msg
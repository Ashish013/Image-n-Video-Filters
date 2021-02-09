import numpy as np
import cv2,math
import streamlit as st
from helper.utils import points2array

def horns_nd_fangs_overlay(input_img,landmark_files):
  msg = None
  detector,predictor = landmark_files
  detections = detector(input_img,1)

  if len(detections) == 0:
    msg = "Face not found. Try re-aligning your face."
    return input_img,msg

  for rect in detections:
    # Warpping Horns
    horns = cv2.imread("./helper/horns.jpg")
    horns = horns[:,:,:3]
    landmarks = predictor(input_img,rect)
    landmarks = points2array(landmarks.parts())

    lower_left = landmarks[1]
    lower_right = landmarks[15]
    lower_slope = ((lower_left[1]-lower_right[1])/( lower_right[0] - lower_left[0]))

    if lower_slope != 0:
      perpendicular_slope = -1/lower_slope
    else:
      perpendicular_slope = 1e-10

    r = int(math.sqrt((landmarks[30][0]-landmarks[8][0])**2 + (landmarks[30][1]-landmarks[8][1])**2))
    theta = math.atan(perpendicular_slope)

    # Due to decrasing y as we go up and x always being < 90 we need to modify the original parametric equation
    #sign_dict = {"x": "+","y": "-"} --> when theta > 0
    #sign_dict = {"x": "-","y": "+"} ---> when theta < 0
    if (theta >= 0):
      upper_left_x = int(lower_left[0] + r * math.cos(theta))
      upper_left_y = int( lower_left[1] - r * math.sin(theta))
      upper_right_x = int(lower_right[0] + r * math.cos(theta) )
      upper_right_y = int(lower_right[1] - r * math.sin(theta))
    else:
      upper_left_x = int(lower_left[0] - r * math.cos(theta))
      upper_left_y = int( lower_left[1] + r * math.sin(theta))
      upper_right_x = int(lower_right[0] - r * math.cos(theta) )
      upper_right_y = int(lower_right[1] + r * math.sin(theta))

    left_upper = [upper_left_x,upper_left_y]
    right_upper = [upper_right_x,upper_right_y]
    right_lower = lower_right
    left_lower = lower_left

    pts_dst = np.array([left_upper,right_upper,right_lower,left_lower])
    y = 800
    pts_src = np.array([(250,395),(780,395),(780,y),(250,y)])

    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]
    horns_warpped = cv2.warpPerspective(horns,H,(input_img.shape[1],input_img.shape[0]))
    horns_bw = cv2.cvtColor(horns_warpped,cv2.COLOR_RGB2GRAY)
    input_img_mask = cv2.threshold(horns_bw,1,255,cv2.THRESH_BINARY_INV)[1]
    masked_input_img = cv2.bitwise_and(input_img,input_img,mask = input_img_mask)

    input_img = cv2.bitwise_or(masked_input_img,horns_warpped)
    #-----------------------------------------------------------------------------------------#
    # Wrapping Fangs
    fangs = cv2.imread("./helper/fangs.jpg")
    fangs = fangs[130:520,105:730,:3]

    upper_left = landmarks[60]
    upper_right = landmarks[64]
    upper_slope = ((lower_left[1]-lower_right[1])/( lower_right[0] - lower_left[0]))

    if upper_slope != 0:
      perpendicular_slope = -1/lower_slope
    else:
      perpendicular_slope = 1e-10

    r = int(math.sqrt((landmarks[62][0]-landmarks[33][0])**2 + (landmarks[62][1]-landmarks[33][1])**2))
    theta = math.atan(perpendicular_slope)

    # Due to decrasing y as we go up and x always being < 90 we need to modify the original parametric equation
    #sign_dict = {"x": "-","y": "+"} --> when theta > 0
    #sign_dict = {"x": "+","y": "-"} ---> when theta < 0
    if (theta >= 0):
      lower_left_x = int(upper_left[0] - r * math.cos(theta))
      lower_left_y = int(upper_left[1] + r * math.sin(theta))
      lower_right_x = int(upper_right[0] - r * math.cos(theta))
      lower_right_y = int(upper_right[1] + r * math.sin(theta))
    else:
      lower_left_x = int(upper_left[0] + r * math.cos(theta))
      lower_left_y = int(upper_left[1] - r * math.sin(theta))
      lower_right_x = int(upper_right[0] + r * math.cos(theta))
      lower_right_y = int(upper_right[1] - r * math.sin(theta))

    left_lower = [lower_left_x,lower_left_y]
    right_lower = [lower_right_x,lower_right_y]
    right_upper = upper_right
    left_upper = upper_left

    pts_dst = np.array([left_upper,right_upper,right_lower,left_lower])
    pts_src = np.asarray(([(0,0),(fangs.shape[1],0),(fangs.shape[1],fangs.shape[0]),(0,fangs.shape[0])]))

    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]
    fangs_warpped = cv2.warpPerspective(fangs,H,(input_img.shape[1],input_img.shape[0]))

    input_img = cv2.bitwise_or(fangs_warpped,input_img)

  return input_img,msg
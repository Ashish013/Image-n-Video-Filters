import cv2,os,math
import numpy as np
from helper.utils import points2array

def overlay_moustache(input_img,landmark_files,index):
  detector,predictor = landmark_files
  msg = None

  if index == 1:
    moustache = cv2.imread("./helper/moustache1.jpg")
    moustache = moustache[185:310,140:500,:3]
  elif index == 2:
    moustache = cv2.imread("./helper/moustache2.jpg")
    moustache = moustache[68:215,:,:3]
  moustache_inv = cv2.bitwise_not(moustache)

  detections = detector(input_img,1)
  if len(detections) == 0:
    msg = "Face not found. Try re-aligning your face."
    return input_img,msg

  for rect in detections:
    landmarks = predictor(input_img,rect)
    landmarks = points2array(landmarks.parts())

    lower_left = [(landmarks[48][0] + landmarks[3][0])//2,(landmarks[48][1] + landmarks[3][1])//2]
    lower_right = [(landmarks[54][0] + landmarks[13][0])//2,(landmarks[54][1] + landmarks[13][1])//2]
    lower_slope = ((lower_left[1]-lower_right[1])/( lower_right[0] - lower_left[0]))

    if lower_slope != 0:
      perpendicular_slope = -1/lower_slope
    else:
      perpendicular_slope = 1e-10

    r = int(math.sqrt((landmarks[33][0]-landmarks[62][0])**2 + (landmarks[33][1]-landmarks[62][1])**2))
    theta = math.atan(perpendicular_slope)

    # Due to decrasing y as we go up and x always being < 90 we need to modify the original paramnetric equation
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

    pts_src = np.asarray(([(0,0),(moustache.shape[1],0),(moustache.shape[1],moustache.shape[0]),(0,moustache.shape[0])]))
    pts_dst = np.array([left_upper,right_upper,right_lower,left_lower])

    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]
    wrapped_overlay_inv = cv2.warpPerspective(moustache_inv,H,(input_img.shape[1],input_img.shape[0]))
    wrapped_overlay = cv2.bitwise_not(wrapped_overlay_inv)
    input_img = cv2.bitwise_and(wrapped_overlay,input_img)

  return input_img,msg
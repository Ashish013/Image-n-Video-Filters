import cv2
import numpy as np
from helper.utils import points2array

def overlay_thuglife(input_img,landmark_files):

  detector,predictor = landmark_files
  msg = None

  # Loads the spectacles image and crops it
  specs = np.array(cv2.imread("helper/specs.jpg"),dtype = np.uint8)
  specs_crop = specs[420:498,179:567,:3]
  inv_specs_crop = cv2.bitwise_not(specs_crop)

  # Loads the cigar image
  cigar = np.array(cv2.imread("helper/cigar.jpg"),dtype = np.uint8)

  # Initializes the face detector and landmark predictor
  detections = detector(input_img,1)

  if len(detections) == 0:
    msg = "Face not found. Try re-aligning your face."
    return input_img,msg

  for rect in detections:
    # Facial landmarks are identified and reformatted into an array type
    landmarks = predictor(input_img,rect)
    landmarks = points2array(landmarks.parts())

    # Calculating spectacles landmarks locations
    specs_left = landmarks[0]
    specs_right = landmarks[16]

    # Eye width is used to tweak the spectacle land mark positions calculated below.
    eyewidth = max(landmarks[40][1] - landmarks[38][1],landmarks[44][1] - landmarks[46][1])

    specs_leftup = [specs_left[0],int(specs_left[1] - 1.5 * eyewidth)]
    specs_rightup = [specs_right[0],int(specs_right[1] - 1.5 * eyewidth)]

    specs_leftdown = [specs_left[0],int(specs_left[1] + 1.5 * eyewidth)]
    specs_rightdown = [specs_right[0],int(specs_right[1] + 1.5 * eyewidth)]

    pts_src = np.array([(0,0),(specs_crop.shape[1]-1,0),(specs_crop.shape[1]-1,specs_crop.shape[0]-1),(0,specs_crop.shape[0]-1)])
    pts_dst = np.asarray([specs_leftup,specs_rightup,specs_rightdown,specs_leftdown])
    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]

    # Spectacles are wrapped on to the input image
    specs_mask_inv = cv2.warpPerspective(inv_specs_crop,H,(input_img.shape[1],input_img.shape[0]))
    specs_mask = cv2.bitwise_not(specs_mask_inv)
    input_img = cv2.bitwise_and(specs_mask,input_img)

    #Calculating cigar landmark locations
    cigar_leftup = landmarks[62]
    cigar_leftdown = landmarks[57]
    cigar_rightup = [landmarks[13][0],cigar_leftup[1]]
    cigar_rightdown = [landmarks[13][0],cigar_leftdown[1]]

    pts_src = np.array([(0,0),(cigar.shape[1]-1,0),(cigar.shape[1]-1,cigar.shape[0]-1),(0,cigar.shape[0]-1)])
    pts_dst = np.array([cigar_leftup,cigar_rightup,cigar_rightdown,cigar_leftdown])
    H = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC)[0]

    # Cigar is wrapped on to the input image
    cigar_out = cv2.warpPerspective(cigar,H,(input_img.shape[1],input_img.shape[0]))
    cigar_out_gray = cv2.cvtColor(cigar_out,cv2.COLOR_RGB2GRAY)
    cigar_mask_inv = cv2.threshold(cigar_out_gray,0,255,cv2.THRESH_BINARY)[1]
    cigar_mask = cv2.bitwise_not(cigar_mask_inv)

    # Final output image is calculated
    mask = cv2.bitwise_and(input_img,input_img,mask = cigar_mask)
    input_img = cv2.bitwise_or(mask,cigar_out)

  return input_img,msg

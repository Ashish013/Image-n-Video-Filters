import numpy as np
import cv2
from helper.utils import points2array

def blur_all_faces(input_img,landmark_files):
	detector,predictor = landmark_files
	msg = None
	detections = detector(input_img,1)
	if len(detections) == 0:
		msg = "Face not found. Try re-aligning your face."
		return input_img,msg

	canvas = input_img.copy()
	for rect in detections:
	  landmarks = predictor(input_img,rect)
	  landmarks = points2array(landmarks.parts())
	  indices = list(range(16)) + [26,25,24,19,18,17,0]
	  pts = np.array(landmarks)[indices].reshape(-1,1,2)
	  canvas = cv2.fillPoly(canvas,[pts],(255,255,255))
	  canvas = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
	  face_mask = cv2.threshold(canvas,254,255,0,cv2.THRESH_BINARY)[1]
	  face_mask_inv = cv2.bitwise_not(face_mask)
	  blur_face = cv2.GaussianBlur(input_img,(37,37),150)
	  blur_face = cv2.bitwise_and(blur_face,blur_face,mask = face_mask)
	  bg = cv2.bitwise_and(input_img,input_img,mask = face_mask_inv)
	  input_img = cv2.bitwise_or(bg,blur_face)

	return input_img,msg
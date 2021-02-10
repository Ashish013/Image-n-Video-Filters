import cv2
import numpy as np
import streamlit as st

def edged_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

def color_quantization(img, k):
# Transform the image
  data = np.float32(img).reshape((-1, 3))
# Determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# Implementing K-Means
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

def cartoonify(img,params = []):
    if params == []:
        line_size,blur_value,total_color = 7,7,11
    else:
        line_size,blur_value,total_color = params

    edges = edged_mask(img, line_size, blur_value)
    quantized_img = color_quantization(img, total_color)
    blurred = cv2.bilateralFilter(quantized_img, d=7, sigmaColor=200,sigmaSpace=200)
    cartooned_img = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartooned_img

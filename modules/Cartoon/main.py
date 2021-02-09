import cv2
import numpy as np
import streamlit as st

def cartoonify(input_img,params = []):
    if params == []:
        #Default parameters
        d,sigmaValue,blockSize,c = 30,300,13,5
    else:
        d,sigmaValue,blockSize,c = params

    img_color = cv2.bilateralFilter(input_img, d, sigmaValue,sigmaValue)

    # prepare edges
    img_edges = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.adaptiveThreshold(cv2.medianBlur(img_edges,17),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,c)
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
    # combine color and edges
    final_img = cv2.bitwise_and(img_color, img_edges)
    #kernel = np.ones((7,7),dtype = np.uint8)
    #final_img = cv2.morphologyEx(final_img,cv2.MORPH_CLOSE,kernel)
    return final_img

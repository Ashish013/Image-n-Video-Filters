import cv2
import numpy as np
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

def generate_ssim_mask(frame,bg,ksize=4):
    
    fg = frame
        
    fg_gray = cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)

    (score,diff) = structural_similarity(fg_gray,bg_gray,full = True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    canvas = np.zeros_like(thresh)
    
    contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    try:
        cnt = contours[0]
    except:
        return thresh
    
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    
    cv2.drawContours(canvas,[cnt],-1, (255, 255, 255),cv2.FILLED)
    kernel = np.ones((int(ksize),int(ksize)),dtype = np.uint8)
    cv2.dilate(canvas,kernel,iterations = 10)
    
    contours,_ = cv2.findContours(canvas,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(contours[0])
    
    return cv2.drawContours(canvas,[hull],-1, (255, 255, 255),cv2.FILLED)

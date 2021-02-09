import cv2,time
import numpy as np
from modules.TimeFreeze.ssim_utils import generate_ssim_mask
from helper.rcnn_utils import generate_rcnn_mask

def time_freeze(frame,bg,params,method = "ssim"):

    stitched_img,prev_num,first_snap,first_frame,start_time = params
    # Important args used in the script
    buffer_time = 5
    font = cv2.FONT_HERSHEY_COMPLEX
    rcnn_file_path = "./helper/rcnn_files"
    # Offset position of the timer from the ends of the frame
    offset = 75

    if(first_snap == False):
        stitched_img = frame

    if(method == "ssim"):
        if(first_frame == False):
            # Captures the static background in the first frame,
            # which is used later for computing ssim
            bg = frame
            first_frame = True

        thresh = generate_ssim_mask(frame,bg)
        inv_thresh = cv2.bitwise_not(thresh)

    elif(method == "rcnn"):
        thresh = generate_rcnn_mask(frame,rcnn_file_path = rcnn_file_path)
        inv_thresh = cv2.bitwise_not(thresh)

    fg_mask = cv2.bitwise_and(frame,frame,mask = thresh)
    bg_mask = cv2.bitwise_and(stitched_img,stitched_img,mask = inv_thresh)

    # The final image after masking is stored in temp which is copied to
    # stitched_img variable after every 'buffer_time' seconds
    temp = cv2.bitwise_or(fg_mask,bg_mask)
    time_diff = int(time.time() - start_time)

    if((time_diff % buffer_time == 0) and time_diff >= prev_num):
        if(first_snap == False):
            first_snap = True
        stitched_img = temp.copy()
        cv2.putText(temp,"Snap !",(temp.shape[1] - offset - 100, offset),fontFace = font,fontScale = 1,color = (255,255,255),thickness = 2)
        prev_num = time_diff+1

    else:
        cv2.putText(temp,str(time_diff % buffer_time),(temp.shape[1] - offset, offset),fontFace = font,fontScale = 1.5,color = (255,255,255),thickness = 2)

    return temp,stitched_img,bg,prev_num,first_snap,first_frame
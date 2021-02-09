import cv2
import numpy as np

def line_freeze(frame,final_array,dimension,variation):

    thickness = 2
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    if variation == "horizontal":
        x = dimension
        if(x >= thickness):
            final_array[:,x-thickness:x] = frame[:,x-thickness:x]
            final_array[:,x:] = frame[:,x:]
        else:
            final_array = frame
        cv2.line(final_array,(x,0),(x,frame.shape[0]),color = (0,255,0),thickness = thickness)
        return final_array

    elif variation == "vertical":
        y = dimension
        if(y >= thickness):
            final_array[y-thickness:y,:] = frame[y-thickness:y,:]
            final_array[y:] = frame[y:]
        else:
            final_array = frame
        cv2.line(final_array,(0,y),(frame.shape[1],y),color = (0,255,0),thickness = thickness)
        return final_array

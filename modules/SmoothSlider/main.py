import cv2
import numpy as np

def half_slide(frame,last_frame,variation = "horizontal"):

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    thickness = 2

    if(variation == 'vertical'):
        dimension = frame_height
        start = int(0.5 * dimension)
    elif(variation == 'horizontal'):
        dimension = frame_width
        start = int(0.65 * dimension)

    if(variation == 'vertical'):
        frame[start+thickness:,:] = last_frame[start:dimension-thickness,:]
    elif(variation == 'horizontal'):
        frame[:,start+thickness:] = last_frame[:,start:dimension-thickness]

    return frame


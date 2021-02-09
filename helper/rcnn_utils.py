import cv2
import numpy as np

def draw_rcnn_mask(frame,boxes,masks,confThreshold, maskThreshold):

    canvas = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
    for i in range(0, boxes.shape[2]):

        # extracting the class ID of the detection along with the confidence of the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confThreshold and classID == 0:

            # scale the bounding box coordinates back relative to size of the frame and
            # compute the width, height of the bounding box

            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, min(startX, W - 1))
            startY = max(0, min(startY, H - 1))
            endX = max(0, min(endX, W - 1))
            endY = max(0, min(endY, H - 1))

            boxW = endX - startX
            boxH = endY - startY

            # extract the mask, resize it to the bounding box and thresholding it to create a binary mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW +1,boxH+1))
            mask = (mask > maskThreshold).astype(np.uint8)

            # extract the ROI of the image
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            box_mask = cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED, cv2.LINE_8, hierarchy, 100)
            canvas[startY:endY+1, startX:endX+1] = box_mask

    return canvas

def generate_rcnn_mask(frame,confThreshold = 0.5, maskThreshold = 0.8,rcnn_file_path = "./helper/rcnn_files"):

    # Load names of classes
    classesFile = f"{rcnn_file_path}/mscoco_labels_names.txt";
    classes = None
    with open(classesFile, 'rt') as f:
       classes = f.read().rstrip('\n').split('\n')

    # Give the textGraph and weight files for the model
    textGraph = f"{rcnn_file_path}/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    modelWeights = f"{rcnn_file_path}/frozen_inference_graph.pb";

    # Load the network
    net = cv2.dnn.readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
    return draw_rcnn_mask(frame,boxes,masks,confThreshold, maskThreshold)

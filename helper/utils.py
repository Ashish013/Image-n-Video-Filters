import os,requests,cv2,shutil
import streamlit as st
import dlib
file_present = True

def rcnn_files_downloader(filename = "./helper/rcnn_files"):

    print("Downloading files for Mask-Rcnn.....")
    file_url = "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
    response = requests.get(file_url,stream = True)
    if (os.path.exists("./helper/file.tar.gz") == False):
        with open("./helper/file.tar.gz","wb") as file:
            for chunk in response.iter_content(chunk_size = 1024):
                if chunk:
                    file.write(chunk)

        shutil.unpack_archive(os.getcwd() + "/helper/file.tar.gz")
        os.rename("mask_rcnn_inception_v2_coco_2018_01_28",f"{filename}")
        os.remove('./helper/file.tar.gz')

    text_file_url = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
    response = requests.get(text_file_url,stream = True)
    if(os.path.exists(f"{filename}/mscoco_labels_names.txt") == False):
        with open(f"{filename}/mscoco_labels_names.txt","wb") as file:
            for chunk in response.iter_content(chunk_size = 128):
                if chunk:
                    file.write(chunk)

    file_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    response = requests.get(file_url)
    if(os.path.exists(f"{filename}/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt") == False):
        with open(f"{filename}/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt","wb") as file:
            for chunk in response.iter_content(chunk_size = 10):
                if chunk:
                    file.write(chunk)
    print("Download Completed !")

def landmarks_file_downloader(link = "https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat",destination = "helper/shape_predictor_68_face_landmarks.dat",chunk_size = 500):
  '''Helper function to download files using url link via the python requests module'''
  response = requests.get(link,stream = True)

  if (os.path.exists(destination) == False):
    with open(destination,"wb") as file:
      for chunk in response.iter_content(chunk_size):
        if chunk:
          file.write(chunk)

def file_checker(transform_type):
    global file_present
    landmark_list = ["Thug Life","Heart Eyes","Moustaches","Face Blur","Devil-ie"]
    rcnn_list = ["Time Freeze - Rcnn","Green Screen","John Cena XD"]
    parameters = []

    if (transform_type in landmark_list):
        if(os.path.exists("./helper/shape_predictor_68_face_landmarks.dat") == False):
            if st.button("Download files to run the filter"):
                text = st.empty()
                text.write("**Downloading files.....**")
                landmarks_file_downloader()
                text.write("**Files Downloaded successfully !**")
                file_present = True
            else:
                file_present = False

        if(file_present == True):
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('./helper/shape_predictor_68_face_landmarks.dat')
            parameters = [detector,predictor]

    elif (transform_type in rcnn_list):
        if (os.path.exists("./helper/rcnn_files") == False):
            if st.button("Download files to run the filter"):
                text = st.empty()
                text.write("**Downloading files..... (It may take quite some time)**")
                rcnn_files_downloader()
                text.write("**Files Downloaded successfully !**")
                file_present = True
            else:
                file_present = False

    else:
        file_present = True

    return file_present,parameters

def points2array(points):
  '''Helper function that converts the co-ordinates from points format of dlib to an array format'''
  landmarks = []
  for i in range(len(points)):
    landmarks.append([points[i].x,points[i].y])
  return landmarks
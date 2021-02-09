import os,cv2
import numpy as np
import streamlit as st
from webcam import webcam
from videos import video_filters
from images import image_transformations

title = st.empty()
title.header("Social Media Filters")
img_extensions = ["jpg","png","jpeg"]
#vid_extensions = ["mp4","avi","mkv"]

def show_info():
    st.subheader('''
    This app is developed and maintained by Ashish Marisetty :heart:
    Entire deployement and configuration of the app is managed through [Streamlit](http://streamlit.io/).
    In case of any discrepancies, feel free to [raise an issue here](https://github.com/Ashish013/Social-Media-Filters-Deploy/issues)
    ''')
    st.subheader (" **Select a configuration mode in the sidebar to start applying the filters !**")
    st.write('''
        The application contains 15+ filters implemented from scratch in python using Deep Learning and Computer Vision libraries.
        Here are a few images/videos made with the app:
        ''')

st.sidebar.subheader("Choose the mode of operation: ")
selected_option = st.sidebar.selectbox("",["Select from below","Image Filters","Video Filters"])

if selected_option == "Image Filters":
    title.header("Image Filters")
    img_operation_mode = st.selectbox("Upload Images from: ",["--  Select from below  --","Local Storage","Take a snap from Webcam"])

    if img_operation_mode == "Local Storage":
        uploaded_file = st.file_uploader("Upload images and videos",type = img_extensions)
        if uploaded_file is not None:
            img_bytes = uploaded_file.read()
            decoded_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
            result = decoded_img

            filter = st.selectbox("Choose an Image filter: ",["Basic Image Editing","Thug Life","Green Screen","Moustaches",\
            "Devil-ie","Heart Eyes","John Cena XD","Cartoonie","Face Blur"],0)
            image_transformations(result,filter,uploaded_file.name.split(".")[-1])

    elif img_operation_mode == "Take a snap from Webcam":
        result = webcam()
        if result is None:
            st.write("Waiting for capture...")
        else:
            st.write("Got an image from the webcam :P")
            result = cv2.cvtColor(np.asarray(result,np.uint8),cv2.COLOR_RGB2BGR)

            filter = st.selectbox("Choose an Image filter: ",["Basic Image Editing","Thug Life","Green Screen","Moustaches",\
            "Devil-ie","Heart Eyes","John Cena XD","Cartoonie","Face Blur"],0)
            image_transformations(result,filter)

    else:
        st.sidebar.markdown('''
        This section contains filters to be applied on images.

        Images can be uploaded either from local storage (or) from your webcamera.
        ''')
        st.sidebar.markdown("Choose the source of image, from the drop list on right :point_right:")
        st.sidebar.subheader("Tips for operating on Image Filters: ")
        st.sidebar.markdown('''
            * Un edited pictures provide the best results.
            * Link to download the edited pictures are at the bottom of the page
            ''')

elif selected_option == "Video Filters":
    title.header("Video Filters")

    transform_type = st.selectbox("Select a Video Filter: ", ["Live Stream","Half Slide - Horizontal","Half Slide - Vertical"\
        ,"Line Freeze - Horizontal","Line Freeze - Vertical","Time Freeze - Ssim","Time Freeze - Rcnn","Heart Eyes","Moustaches",\
        "Face Blur","Devil-ie","Thug Life"],0)

    if (transform_type == "Live Stream"):
        st.sidebar.write("This section contains filters to be applied on live video from web camera")
        st.sidebar.subheader("Tips for operating on Video Filters: ")
        st.sidebar.markdown('''
        * In case the video doesn't start streaming or misfunctions, stop the stream and start again using the buttom below the video player.
        * For capturing a filter result, pause the stream from the media player and take a screenshot :sweat_smile:.
            ''')
    elif transform_type == "Time Freeze - Ssim":
        st.markdown("Pre-requisites of this filter are:")
        st.markdown('''
            * **Static background**.
            * **Allow the filter to capture the background for first few frames.**
            ''')
        st.markdown("Or the results will be inaccurate :grimacing:")

    elif transform_type == "Time Freeze - Rcnn":
        st.markdown("**This filter can only be run locally on a GPU !**")

    video_filters(transform_type)

else:
    show_info()

#-------------------------------------------------------------------------------------------------------
# Like button
st.sidebar.subheader("Love the project, then lmk below :")
st.sidebar.write("")
col1,col2 = st.sidebar.beta_columns([1.7,1])
if os.path.exists("./helper/likes.txt"):
    with open("./helper/likes.txt",'r') as file:
        like_count = int(file.read())
else:
    like_count = 0
liked = col1.button("Loved the project 👍")

if liked == True:
    if os.path.exists("./helper/likes.txt"):
        with open("./helper/likes.txt",'r+') as file:
            like_count = int(file.read())
            file.truncate(0)
            file.seek(0)
            like_count+=1
            file.write(str(like_count))
    else:
        with open("./helper/likes.txt",'w+') as file:
            like_count+=1
            file.write(str(like_count))
#st.markdown("""<style>.css-2trqyj{background-color: rgba(0,0,255,0.6);color: white} </style>""", unsafe_allow_html=True)

if like_count != 0:
    col2.markdown(f"{int(like_count)} :heart:")

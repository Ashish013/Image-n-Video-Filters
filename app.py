import os,cv2
import numpy as np
import streamlit as st
from webcam import webcam
from videos import video_filters
from images import image_transformations

title = st.empty()
title.header("Image and Video Filters")
img_extensions = ["jpg","png","jpeg"]
#vid_extensions = ["mp4","avi","mkv"]
#if os.path.exists("./helper/likes.txt"):
#    os.remove("./helper/likes.txt")

def show_info():
    st.subheader('''
    This app is developed and maintained by Ashish Marisetty :heart:
    Entire deployement and configuration of the app is managed through [Streamlit](http://streamlit.io/).
    In case of any discrepancies, feel free to [raise an issue here](https://github.com/Ashish013/Image-n-Video-Filters/issues).


    *For more interesting projects and explanation of the source code, consider following [my blogs here.](https://ashish013.github.io/opencv)*
    ''')
    st.subheader (" **Select a configuration mode in the sidebar to start applying the filters :point_left:**")
    st.write('''
        The website contains 15+ filters implemented from scratch in python using Deep Learning and Computer Vision libraries.
        Here are a few images/videos made with the website:

        **Note that all edited images below are directly generated without any manual intervention.**
        ''')
    st.markdown("#### **Filters used: Cartoonie, Moustache -2, Thug Life**")
    st.write("")
    col1,col2 = st.beta_columns(2)
    col1.image("./helper/dwayne.jpg",use_column_width = True,clamp = True)
    col2.image("./helper/dwayne_edit.jpg",use_column_width = True,clamp = True)

    st.markdown("#### **Filters used: Line Freeze - Vertical**")
    st.write("")
    st.image("./helper/Line Freeze.gif",use_column_width = True)

    st.markdown("#### **Filters used: Devil-ie, Heart Eyes**")
    st.write("")
    col1,col2 = st.beta_columns(2)
    col1.image("./helper/trump.jpg",use_column_width = True,clamp = True)
    col2.image("./helper/trump_edit.jpg",use_column_width = True,clamp = True)

    st.markdown("#### **Filters used: Half Slide - Vertical**")
    st.write("")
    st.image("./helper/Line Slide.gif",use_column_width = True)

    st.markdown("#### **Filters used: Green Screen**")
    st.write("")
    col1,col2 = st.beta_columns(2)
    col1.image("./helper/elon_kanye.jpg",use_column_width = True,clamp = True)
    col2.image("./helper/elon_kanye_edit.jpeg",use_column_width = True,clamp = True)

    st.markdown("#### **This video can be recreated using Time Freeze - Ssim filter **")
    st.write("")
    st.image("./helper/Time Freeze.gif",use_column_width = True)

    st.write("")

    st.subheader (" **Select a configuration mode in the sidebar to start applying the filters :point_left:**")
    st.write("All the above input images are for education purposes only.")

def like_button(st_object,key_value):
    #-------------------------------------------------------------------------------------------------------
    # Like button

    st_object.subheader("Love the project, then lmk below :")
    st_object.write("")
    col1,col2 = st_object.beta_columns([1.7,1])
    if os.path.exists("./helper/likes.txt"):
        with open("./helper/likes.txt",'r') as file:
            like_count = int(file.read())
    else:
        like_count = 0
    liked = col1.button("Loved the project 👍",key = key_value)

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
    else:
        if os.path.exists("./helper/likes.txt"):
            with open("./helper/likes.txt",'r+') as file:
                like_count = int(file.read())
                file.seek(0)
    #st.markdown("""<style>.css-2trqyj{background-color: rgba(0,0,255,0.6);color: white} </style>""", unsafe_allow_html=True)

    if like_count != 0:
        col2.markdown(f"{int(like_count)} :heart:")
    
    
st.sidebar.subheader("Choose the mode of operation: ")
selected_option = st.sidebar.selectbox("",["Select from below","Image Filters","Video Filters"])

if selected_option == "Image Filters":
    title.header("Image Filters")
    img_operation_mode = st.selectbox("Upload Images from: ",["--  Select from below  --","Local Storage","Take a snap from Webcam"])

    if img_operation_mode == "Local Storage":
        uploaded_file = st.file_uploader("Upload images from local storage here",type = img_extensions)
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
    #st.markdown("""<style>.css-3mnucz a {color: red} </style>""", unsafe_allow_html=True)
    #st.markdown("<span style='color:red'>In case video filters don't work in online mode, follow the instructions in <a href \
    #    = 'https://github.com/Ashish013/Image-n-Video-Filters/blob/main/README.md#how-to-run-the-application-from-a-local-host'>Readme</a>\
    #     to run them on localhost.</span>",unsafe_allow_html=True)
    st.markdown("In case video filters don't work in online mode, follow the instructions in [Readme]\
    (https://github.com/Ashish013/Image-n-Video-Filters/blob/main/README.md#how-to-run-the-application-from-a-local-host) to run them on localhost")
    transform_type = st.selectbox("Select a Video Filter: ", ["Live Stream","Half Slide - Horizontal","Half Slide - Vertical"\
        ,"Line Freeze - Horizontal","Line Freeze - Vertical","Time Freeze - Ssim","Time Freeze - Rcnn","Heart Eyes","Moustaches",\
        "Face Blur","Devil-ie","Thug Life"],0)

    if (transform_type == "Live Stream"):
        st.sidebar.write("This section contains filters to be applied on live video from web camera")
        st.sidebar.subheader("Tips for operating on Video Filters: ")
        st.sidebar.markdown('''
        * In case the video  misfunctions/ glitches, stop the stream and start again using the buttom below the video player.
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

like_button(st.sidebar,"sidebar_button")
like_button(st,"button")

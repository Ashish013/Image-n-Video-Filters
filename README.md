# Image-n-Video-Filters
This repository contains the source code for the [streamlit app hosted here](https://share.streamlit.io/ashish013/image-n-video-filters/main/app.py)
The application contains 15+ filters implemented from scratch in python using Deep Learning and Computer Vision libraries.

## Why the Video filters are not working:
It maybe because there are some problems on the network between your local environment to the remote host (Streamlit sharing server). For example, inside a office network, the FW can be configured to drop some packets (We can use relay servers called TURN server that can solve this problem as Google Meet or Zoom may do, but this an example application and it costs to have such a server)

**The easiest way to get over this problem is conecting from a local host**.

## How to run the application from a local host:
```
git pull https://github.com/Ashish013/Image-n-Video-Filters
pip install -r requirements.txt
python app.py
```

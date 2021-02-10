import cv2
import os
import base64
import numpy as np
import streamlit as st
from PIL import ImageFont,ImageDraw,Image
from modules.ThugLife.main import overlay_thuglife
from modules.Cartoon.main import cartoonify
from modules.HeartEyes.main import overlay_heart_eyes
from modules.Moustaches.main import overlay_moustache
from modules.BlurFaces.main import blur_all_faces
from modules.DevilFace.main import horns_nd_fangs_overlay

from helper.utils import file_checker
from helper.descriptor import filter_info
from helper.rcnn_utils import generate_rcnn_mask

img_extensions = ["jpg","png","jpeg"]

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<h3><a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a></h3>'
    return href

def image_transformations(result,filter,img_extension = "jpg"):

	result = result[:,:,:3]
	file_checker_return = file_checker(filter)
	if file_checker_return[0] == True:

		filter_info(filter)
		if filter == "Basic Image Editing":
			gamma = st.slider("Gamma Correction",0.0,5.0,1.0,step = 0.1)
			saturation = st.slider("Saturation",0.0,2.0,1.0,step = 0.1)

			bluring = st.checkbox("Bluring",False)
			if bluring:
				col1,col2 = st.beta_columns(2)
				blur_area = col1.slider("Blur Area",1,101,1,2)
				blur_intensity = col2.slider("Blur Intensity",0,50,0,1)
				result = cv2.GaussianBlur(result,(blur_area,blur_area),blur_intensity)

			apply_vintage = st.checkbox("Apply Vignette effect",False)
			if apply_vintage:
					vignette_effect = st.slider("Vignette Intensity",1,120,1)
					rows, cols = result.shape[:2]
					# Create a Gaussian filter
					kernel_x = cv2.getGaussianKernel(cols,vignette_effect+139)
					kernel_y = cv2.getGaussianKernel(rows,vignette_effect+139)
					kernel = kernel_y * kernel_x.T
					filter = 255 * kernel / np.linalg.norm(kernel)
					vignette_img = np.copy(result)
					# for each channel in the input image, we will apply the above filter
					for i in range(3):
						vignette_img[:,:,i] = vignette_img[:,:,i] * filter
					result = vignette_img

			hsvImg = cv2.cvtColor(result,cv2.COLOR_BGR2HSV)
			hsvImg[...,1] = np.clip(hsvImg[...,1]*saturation,0,255)
			hsvImg[...,2] = np.power((hsvImg[...,2]/255.0),1/(gamma+0.1)) * 255.0
			result = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)

		elif filter == "Thug Life":
			txt = "Thug Life "
			fontsize = 1  # starting font size
			img_fraction = 0.50 # portion of img the text should cover
			h,w = result.shape[0],result.shape[1]
			font_path = "./helper/Killig.ttf"
			font = ImageFont.truetype(font_path, fontsize)
			while font.getsize(txt)[0] < img_fraction*w:
				# iterate until the text size is just larger than the criteria
				fontsize += 1
				font = ImageFont.truetype(font_path, fontsize)

			result,msg = overlay_thuglife(result,file_checker_return[1])
			if msg != None:
				st.write(msg)
			else:
				pil_img = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB) )
				draw = ImageDraw.Draw(pil_img)
				y_offset = int(0.17 * h)
				draw.text((w//2,h - y_offset),txt,font = font)
				result = cv2.cvtColor(np.asarray(pil_img,dtype = np.uint8),cv2.COLOR_RGB2BGR)

		elif filter == "John Cena XD":
			mask = generate_rcnn_mask(result,0.3,0.3)
			if np.all(mask != np.zeros_like(mask)):
				txt = "JUST ME AND JOHN CENA CHILLING..."
			else:
				txt = "HEY LOOK ITS JOHN CENA THERE !!"

			fontsize = 1  # starting font size
			font_path = "./helper/Helvetica-Bold.ttf"
			img_fraction = 1 # portion of img the text should cover
			h,w = result.shape[0],result.shape[1]
			y_offset = int(0.1 * h)
			result[h-y_offset:] = (0,0,0)
			font = ImageFont.truetype(font_path, fontsize)
			while font.getsize(txt)[0] <= img_fraction*w:
				# iterate until the text size is just larger than the criteria
				fontsize += 1
				font = ImageFont.truetype(font_path, fontsize)
			pil_img = Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB) )
			draw = ImageDraw.Draw(pil_img)
			draw.text((0,h - y_offset),txt,font = font)
			result = cv2.cvtColor(np.asarray(pil_img,dtype = np.uint8),cv2.COLOR_RGB2BGR)

		elif filter == "Cartoonie":
			lineSize = st.slider("Number of edges",3,101,7,2)
			blurValue = st.slider("Blur effect",3,101,7,2)
			totalColors = st.slider("Total number of colors in image",2,100,12,1)
			result = cartoonify(result,[lineSize,blurValue,totalColors])

		elif filter == "Heart Eyes":
			result,msg = overlay_heart_eyes(result,file_checker_return[1])
			if msg != None:
				st.write(msg)

		elif filter == "Moustaches":
			selected_style = st.select_slider("",["Style 1","Style 2"])
			result,msg = overlay_moustache(result,file_checker_return[1],int(selected_style.split()[-1]))
			if msg != None:
				st.write(msg)

		elif filter == "Face Blur":
			result,msg = blur_all_faces(result,file_checker_return[1])
			if msg != None:
				st.write(msg)

		elif filter == "Devil-ie":
			result,msg = horns_nd_fangs_overlay(result,file_checker_return[1])
			if msg != None:
				st.write(msg)

		elif filter == "Green Screen":
			st.write("")
			st.write("**Upload Background Image:**")
			bg_upload = st.file_uploader("",type = img_extensions)
			if bg_upload is not None:
				img_bytes = bg_upload.read()
				bg = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
				confThresh = st.slider("Confidence Threshold",0.0,1.0,0.8,0.01)
				maskThresh = st.slider("Mask Threshold",0.0,1.0,0.35,0.01)
				tempt_text = st.empty()
				tempt_text.write("Processing....")
				mask = generate_rcnn_mask(result,confThresh,maskThresh)
				tempt_text.write("")
				if np.all(mask == np.zeros_like(mask)):
					st.write("Person not detected to overlay as foregreound !")
				fg = cv2.bitwise_and(result,result,mask = mask)
				bg = cv2.resize(bg,(fg.shape[1],fg.shape[0]))
				bg = cv2.bitwise_and(bg,bg,mask = cv2.bitwise_not(mask))
				result = cv2.bitwise_or(fg,bg)
			else:
				result = None

		if np.all(result != None):
			st.image(result,use_column_width = True,clamp = True,channels = "BGR")
			filename = "Output" + "." + img_extension
			cv2.imwrite(filename, result)
			st.markdown(get_binary_file_downloader_html(filename, 'the edited picture,by right clicking and select "Save the link as":P'), unsafe_allow_html=True)
			st.markdown("Above link :point_up: is only applicable to PC's")

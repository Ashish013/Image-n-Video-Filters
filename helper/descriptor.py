import streamlit as st

def filter_info(filter):

	if filter != "Live Stream":
		st.sidebar.subheader("Filter Information: ")
	if filter == "Basic Image Editing":
		st.sidebar.write('''
			* Gamma Correction can be used to approximate brigtness in an image.")
			* Saturation can be used to control saturation in an image.
			* Bluring enables bluring of images using a Gaussian filter.
			''')
	elif filter == "Thug Life":
		st.sidebar.write('''
			* Thug Life spectacles and cigar is automatically applied on to the face.
			* The filter is applied by identifying 68 facial landmarks, computing the orientation of\
			 face and finally warpping the overlay on to the face.
			''')
	elif filter == "Green Screen":
		st.sidebar.write('''
			* First upload the image to be used as foreground image.
			* Next upload the image to be used as background image.
			* The filter identifies the person in foreground image using Deep Learning model (Mask-RCNN)\
			 and passes this mask to be applied as foreground on to the background image.
			* This produces a Green screen effect without an actual one.
			''')
	elif filter == "Moustaches":
		st.sidebar.write('''
			* Moustache is automatically applied on to the face.
			* There are 2 different options of moustaches to select from.
			* The filter is applied by identifying 68 facial landmarks, computing the orientation of\
			 face and finally warpping the overlay on to the face.
			''')
	elif filter == "Devil-ie":
		st.sidebar.write('''
			* Devil horns and fangs are automatically applied on to the face.
			* The filter is applied by identifying 68 facial landmarks, computing the orientation of\
			 face and finally warpping the overlay on to the face.
			''')
	elif filter == "Heart Eyes":
		st.sidebar.write('''
			* Heart Eyes are automatically applied on to the face.
			* The filter is applied by identifying 68 facial landmarks, computing the orientation of\
			 face and finally warpping the overlay on to the face.
			''')
	elif filter == "John Cena XD":
		st.sidebar.write(" ")
		st.sidebar.image("./helper/john_cena.jpg",use_column_width = True,clamp = True)

	elif filter == "Cartoonie":
		st.sidebar.write('''
			* Cartoonie tranforms the image into a cartoon-ie and sketch-y kinda look.
			* Cartoon Effect slider allows you to control the effect applied on the image.
			* Higher the Edge Controller value, more prominent the edges in the image.
			''')
	elif filter == "Face Blur":
		st.sidebar.write('''
			* Face Blur identifies all the faces in the image and blurs them.
			* The filter works by computing the key facial features and bluring only them to help retian other features of the face.
			''')
	elif filter == "Half Slide - Horizontal":
		st.sidebar.write('''
			* Half Silde- Horizontal freezes the left part of the frame, where as producing a sliding effect on the other part of the frame.
			''')
	elif filter == "Half Slide - Vertical":
		st.sidebar.write('''
			* Half Silde- Horizontal freezes the top part of the frame, where as producing a sliding effect on the other part of the frame.
			''')
	elif filter == "Line Freeze - Horizontal":
		st.sidebar.write('''
			* A green line passes through the frame from left to right.
			* Line Freeze - Horizontal freezes the part of the frame that the line has passed through, while keeping the other side o the line\
			intact.
			''')
	elif filter == "Line Freeze - Vertical":
		st.sidebar.write('''
			* A green line passes through the frame from top to bottom.
			* Line Freeze - Horizontal freezes the part of the frame that the line has passed through, while keeping the other side o the line\
			intact.
			''')
	elif filter == "Time Freeze - Ssim":
		st.sidebar.write('''
			* Time Freeze - Ssim captures the persons instance and pose every 5 seconds.
			* This captured frame is overlayed on the upcoming frames and after 5 seconds another instance of the person is captured and overlayed.
			* And the cycle repeats...
			* The filter is implemented using the Structural Similarity Index method and hence requires a static background.
			''')
	elif filter == "Time Freeze - Rcnn":
		st.sidebar.write('''
			* Time Freeze - Ssim captures the persons instance and pose every 5 seconds.
			* This captured frame is overlayed on the upcoming frames and the cycle repeats...
			* The filter calculates the person's instance using a Deep Learning model (Mask - RCNN).
			''')
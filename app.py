"""
Interferometry demo.

Video capture only works locally.
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas
import sys
import pdb

from utils import *


if __name__ == "__main__":

	st.set_page_config(layout="wide")

	col1, col2, col3 = st.columns(3)

	with col1:
		# array layout
		st.title("Array Layout")
		c1, c2, c3 = st.columns(3)
		with c1:
			# load antennas
			option = st.selectbox('Array type:',
			    ('Uniform', 'Gaussian', 'Hexagon', 'Grid', 'Upload', 'Capture'))
		with c2:
			seed = st.number_input("Random Seed:", value=0, min_value=0)
			np.random.seed(seed)
		with c3:
			Nant = st.number_input("Number of Antennas:", min_value=2, max_value=500, value=50)

		if option == 'Uniform':
			ants = np.vstack([np.random.rand(500)*4/3/2, np.random.rand(500)/2])
			ants = ants[:, :Nant]
		elif option == 'Gaussian':
			ants = np.vstack([np.random.randn(500)/6*4/3, np.random.randn(500)/6])
			ants = ants[:, :Nant]
		elif option == 'Grid':
			Nant = min([Nant, 25])
			ants = np.vstack([np.meshgrid(np.linspace(0, .5, Nant), np.linspace(0, .5, Nant), indexing='ij')]).reshape(2, -1)
		elif option == 'Hexagon':
			Nants = min([Nant, 15])
			ants = make_hex(Nants, D=.3 / Nants)
		elif option in ['Upload', 'Capture']:
			if option == 'Upload':
				uploaded_file = st.file_uploader('Upload a white-background image:')
			else:
				uploaded_file = st.camera_input('Take a white-background photo:')
			if uploaded_file is None: sys.exit()
			file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
			# get grayscale
			img = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2GRAY)
			# slight denoising
			blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
			# adaptive threshold
			thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
			# contour
			contours, hiers = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# get contour centers
			ants = []
			for cont in contours:
				M = cv2.moments(cont)
				if M['m00'] > 0:
					cx = M['m10']/M['m00']
					cy = img.shape[0] - M['m01']/M['m00']
					ants.append(np.array([cx, cy]))

			ants = np.array(ants).T[:, :500]
			ants /= (ants[0].max() - ants[0].min()) * 2

			# plot segments
			st.image(255-thresh, clamp=True, width=400)
			st.write("Nantenna: {}".format(ants.shape[1]))

		if option not in ['Upload', 'Capture']:
			st.scatter_chart(pandas.DataFrame(ants.T, columns=['X', 'Y']), x='X', y='Y',
							 height=400, width=400, use_container_width=False, size=30, color='#000000')

	with col2:
		st.title("UV Sampling")

		c1, c2 = st.columns(2)
		with c1:
			# get rotation
			rotate = st.number_input("Earth Rotation (degrees):", min_value=0, max_value=90, value=0, step=5)
		with c2:
			scale = st.number_input("UV Scale:", min_value=.04, max_value=2., value=1., step=.05)

		# get UV object
		UV = UVSample(ants * scale, rotate=rotate, Nu=256, ratio=3/4, keep_zero=True)

		# plot UV coverage
		uvs = UV.uvs
		if uvs.shape[1] > 10000:
			uvs = uvs[:, ::uvs.shape[1]//10000]
		st.scatter_chart(pandas.DataFrame(uvs.T, columns=['U', 'V']), x='U', y='V',
						 height=600, width=300, use_container_width=False, size=15)


	with col3:
		st.title('Original & Reconstructed')


		option = st.selectbox('Image:',
		    ('Galaxies', 'Pleiades', 'Black Hole', 'Point', 'Upload', 'Capture'))

		if option == 'Galaxies':
			IM = Image(cv2.imread('data/deepfield.jpeg')[..., ::-1], shape=UV.shape)
		elif option == 'Pleiades':
			IM = Image(cv2.imread('data/pleiades.jpg')[..., ::-1], shape=UV.shape)
		elif option == 'Point':
			IM = Image(cv2.imread('data/point_source.png'), shape=UV.shape)
		elif option == 'Black Hole':
			IM = Image(cv2.imread('data/black_hole.webp'), shape=UV.shape)
		elif option in ['Upload', 'Capture']:
			if option == 'Upload':
				uploaded_file = st.file_uploader('Upload an image:', label_visibility='collapsed')
			else:
				uploaded_file = st.camera_input('Take a white-background photo:')
			if uploaded_file is None: sys.exit()
			file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
			IM = Image(cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB), shape=UV.shape)

		# plot original image
		if option != 'Capture':
			st.image(IM.img, clamp=True, width=400)
		st.image(UV(IM.img)[0], clamp=True, width=400)


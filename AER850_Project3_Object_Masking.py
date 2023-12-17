#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:05:55 2023

AER850 Project 3 

The purpose of this program is to use OpenCV to extract the motherboard image
using thresholding in conjunction with edge detection techniques.

@author: Akus Chhabra
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the Image and Apply Grayscaling
image_org = cv2.imread('motherboard_image.JPEG', cv2.COLOR_BGR2RGB)
image_rgb = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)

# Set threshold
threshold_value = 120 # 80

# Apply Thresholding
ret, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Creater Inverted Mask
invmask = cv2.bitwise_not(binary_image)

# Find Contours
contours, opt2 = cv2.findContours(invmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Locate maximum contour
max_contour = max(contours, key=cv2.contourArea)
contour_img = image.copy()

# Apply masking
mask = np.zeros_like(image)

# Draw Contours
cv2.drawContours(mask, [max_contour], -1, (255,255,255), thickness = cv2.FILLED)
cv2.drawContours(contour_img, [max_contour], -1, (255,255,255),2)

# Obtain extracted image
extracted = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

#Plot figures
plt.figure(figsize = (100,100))
plt.imshow(contour_img, cmap='gray')

plt.figure(figsize = (100,100))
plt.imshow(mask, cmap='gray')

plt.figure(figsize = (100,100))
plt.imshow(extracted, cmap='gray')
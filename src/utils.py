# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from google.colab.patches import cv2_imshow

def show_img(img, seg_img):
  """
  Helper function to visualize an image and the corresponding segmented image
  """
  plt.figure(figsize = (20, 20))
  plt.subplot(2, 1, 1)
  plt.imshow(img)
  plt.title('Image')

  plt.subplot(2, 1, 2)
  plt.imshow(seg_img, cmap = 'gray')
  plt.title('Segmented image')

  plt.show()

def get_seg_img(seg_img):
  """
  Helper function that maps the segmented image annotation into 0 and 1;

  The initial segmented images have labels 0 and 24 because of the Pixel
  Annotation tool used
  The initial RGG image is converted into a single channel image (gray image)
  of zeros and ones;
  """

  final_seg_img = np.zeros((512, 512, 1)) # initialize the segmented image with zeros

  seg_img = cv2.resize(seg_img, (512, 512))

  seg_img = seg_img[:, :, 0]

  final_seg_img[:, :, 0] = (seg_img != 0).astype(int)

  return final_seg_img

def img_preprocessing(img):
  ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
  img = cv2.resize(img,(512,512))
  img = np.expand_dims(img,axis=-1)
  img = img/255

  return img

def get_bounding_boxes(orig_img, seg_img):
  """
  Inputs: original image and segmented image
  Otput: original image with bounding boxes around the segmented proposed regions
  """
  # scale original image
  orig_img = cv2.resize(orig_img, (512, 512))

  # cast the seg_img to uint8
  seg_img = cv2.convertScaleAbs(seg_img)

  # transform segmented image to the threshold verison using openCV
  ret, seg_img = cv2.threshold(seg_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  
  # get contours around the segmented regions
  contours, hier = cv2.findContours(seg_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  for c in contours:
    x, y, w, h = cv2.boundingRect(c) # get bounding box coordinates

    orig_img = cv2.rectangle(orig_img, (x, y), (x+w,y+h), (0,255,0), 2) # draw bounding box on original image


  return orig_img


      

























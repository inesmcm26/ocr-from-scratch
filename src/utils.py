# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from google.colab.patches import cv2_imshow
import math

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

  seg_img = seg_img[:, :, 0] # get the first channel only

  final_seg_img[:, :, 0] = (seg_img != 0).astype(int)

  return final_seg_img

def img_preprocessing(img):
  ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
  img = cv2.resize(img,(512,512))
  img = np.expand_dims(img,axis=-1)
  img = img/255

  return img

def get_line_segments(orig_path, orig_img_name, model):
  """
  Inputs: path to original image and original image name
  Output: crops of the original image containing the segmented lines
  """

  # read original image and apply preprocessing steps
  img_orig = cv2.imread(f'{orig_path}{orig_img_name}',0)
  img = img_preprocessing(img_orig)
  img = np.expand_dims(img,axis=0)

  # predict the segmented image
  pred = model.predict(img)
  pred = np.squeeze(np.squeeze(pred,axis=0),axis=-1)

  # cast the seg_img to uint8
  pred = cv2.convertScaleAbs(pred)

  coordinates=[]

  img = cv2.normalize(src = pred, dst=None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)

  # transform segmented image to the threshold version using openCV
  # ret, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  (H, W) = img_orig.shape[:2]
  (newW, newH) = (512, 512)
  rW = W / float(newW)
  rH = H / float(newH)

  coordinates = []
  line_array = []
  
  # get contours around the segmented regions
  contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  for c in contours:
    x, y, w, h = cv2.boundingRect(c) # get bounding box coordinates

    # draw bounding box on original image
    #img_orig = cv2.rectangle(img_orig, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (0,255,0), 2)
    
    coordinates.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))

  # Crop the bounding boxes generated from the images to get the segmented lines
  for i in range(len(coordinates)-1,-1,-1):
      coords = coordinates[i]

      p_img = img_orig[coords[1]:coords[3], coords[0]:coords[2]].copy()

      cv2_imshow(p_img)

      line_array.append(p_img)

  return line_array

def get_word_segments(orig_path, orig_img_name, model):
  """
  Inputs: path to original image and original image name
  Output: crops of the original image containing the segmented words
  """

  # read original image and apply preprocessing steps
  img_orig = cv2.imread(f'{orig_path}{orig_img_name}.jpg',0)
  img_pad = pad_image(img_orig)
  img = img_preprocessing(img_pad)
  img = np.expand_dims(img,axis=0)

  # predict the segmented image
  pred = model.predict(img)
  pred = np.squeeze(np.squeeze(pred,axis=0),axis=-1)

  # cast the seg_img to uint8
  plt.imsave('test_img_mask.JPG', pred)

  pred = cv2.imread('/content/test_img_mask.JPG',0)

  ret, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  # transform segmented image to the threshold verison using openCV
  (H, W) = img_pad.shape[:2]
  (newW, newH) = (512, 512)
  rW = W / float(newW)
  rH = H / float(newH)

  coordinates = []
  line_array = []
  
  # get contours around the segmented regions
  contours, hier = cv2.findContours(pred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  for c in contours:
    x, y, w, h = cv2.boundingRect(c) # get bounding box coordinates
    print(x, y, w, h)

    # draw bounding box on original image
    img_pad = cv2.rectangle(img_pad, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (0,255,0), 2)
    coordinates.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))

  # Crop the bounding boxes generated from the images to get the segmented lines
  # for i in range(len(coordinates)-1,-1,-1):
  #     coords = coordinates[i]

  #     p_img = img_orig[coords[1]:coords[3], coords[0]:coords[2]].copy()

  #     cv2_imshow(p_img)

  #     line_array.append(p_img)

  return line_array

# def pad_image(img, padding_value):
#   old_h, old_w = img.shape
#   new_height = max(old_h, 512)
#   to_pad_height = np.ones((new_height - old_h, old_w)) * padding_value
#   if to_pad_height.size:
#       start_height = (new_height - old_h) // 2
#       img = np.concatenate((to_pad_height[:start_height, :], img, to_pad_height[start_height:, :]), axis=0)

#   new_width = max(old_w, 512)
#   to_pad_width = np.ones((new_height, new_width - old_w)) * padding_value
#   if to_pad_width.size:
#       start_width = (new_width - old_w) // 2
#       img = np.concatenate((to_pad_width[:, :start_width], img, to_pad_width[:, start_width:]), axis=1)

#   return img

def pad_img(img):
	old_h,old_w=img.shape[0],img.shape[1]

	#Pad the height.

	#If height is less than 512 then pad to 512
	if old_h<512:
		to_pad=np.ones((512-old_h,old_w))*255
		img=np.concatenate((img,to_pad))
		new_height=512
	else:
	#If height >512 then pad to nearest 10.
		to_pad=np.ones((roundup(old_h)-old_h,old_w))*255
		img=np.concatenate((img,to_pad))
		new_height=roundup(old_h)

	#Pad the width.
	if old_w<512:
		to_pad=np.ones((new_height,512-old_w))*255
		img=np.concatenate((img,to_pad),axis=1)
		new_width=512
	else:
		to_pad=np.ones((new_height,roundup(old_w)-old_w))*255
		img=np.concatenate((img,to_pad),axis=1)
		new_width=roundup(old_w)-old_w
	return img


def pad_seg(img):
	old_h,old_w=img.shape[0],img.shape[1]

	#Pad the height.

	#If height is less than 512 then pad to 512
	if old_h<512:
		to_pad=np.zeros((512-old_h,old_w))
		img=np.concatenate((img,to_pad))
		new_height=512
	else:
	#If height >512 then pad to nearest 10.
		to_pad=np.zeros((roundup(old_h)-old_h,old_w))
		img=np.concatenate((img,to_pad))
		new_height=roundup(old_h)

	#Pad the width.
	if old_w<512:
		to_pad=np.zeros((new_height,512-old_w))
		img=np.concatenate((img,to_pad),axis=1)
		new_width=512
	else:
		to_pad=np.zeros((new_height,roundup(old_w)-old_w))
		img=np.concatenate((img,to_pad),axis=1)
		new_width=roundup(old_w)-old_w
	return img

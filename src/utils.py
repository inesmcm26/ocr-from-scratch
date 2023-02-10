# imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn import train_test_split

def show_img(img, seg_img):
  """
  Helper function to visualize an image and the corresponding segmented image
  """
  plt.figure(figsize = (10, 10))
  plt.subplot(2, 1, 1)
  plt.imshow(img)
  plt.title('Image')

  plt.subplot(2, 1, 2)
  plt.imshow(seg_img)
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

def batch_generator(img_list, seg_list, img_path, seg_path, n_classes, batch_size):
  while True:
    X = []
    Y = []

    for i in range(batch_size):
      idx = random.choice(len(img_list))

      img = cv2.imread(f'{img_path}{img_list[idx]}.JPG', 0)

      # ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

      img = cv2.resize(img, (512, 512))
      img = np.expand_dims(img, axis = -1)
      img = img / 255

      seg = cv2.imread(f'{seg_path}{seg_list[idx]}.png', 1)
      seg = get_seg_img(seg)

      X.append(img)
      Y.append(seg)


    # suspends the function execution and returns value to the caller
    yield np.array(X),np.array(Y)




























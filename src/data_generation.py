import numpy as np
from google.colab.patches import cv2_imshow
import tensorflow as tf
import utils
import cv2
import matplotlib.pyplot as plt

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class LineDataGenerator(tf.keras.utils.Sequence):
  """
  Class that generates batches of the line segmented data
  """
  
  def __init__(self, in_folder, out_folder, list_IDs, seg_IDs, batch_size = 4, dim = (512, 512), n_channels = 1, n_classes = 2, shuffle = True):
      self.in_folder = in_folder
      self.out_folder = out_folder
      self.list_IDs = list_IDs
      self.seg_IDs = seg_IDs
      self.batch_size = batch_size
      self.dim = dim
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.shuffle = shuffle
      self.on_epoch_end()

  def on_epoch_end(self):
    """
    Updates indexes after each ephoc
    """
    self.indexes = np.arange(len(self.list_IDs))

    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __data_generation(self, list_IDs_temp, seg_IDs_temp):
    """
    Generates data containing batch_size samples
    This code is multi-core friendly
    """

    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size, *self.dim, self.n_channels))

    for i, ID in enumerate(list_IDs_temp):
      img = cv2.imread(f'{self.in_folder}{ID}.JPG',0)
      img = utils.img_preprocessing(img)
      X[i,] = img
    
    for i, seg_ID in enumerate(seg_IDs_temp):
      seg = cv2.imread(f'{self.out_folder}{seg_ID}.png',1)
      seg = utils.get_seg_img(seg)
      y[i,] = seg

    return X, y

  def __getitem__(self, index):
    """
    Generates one batch of data
    """

    # Gets images indexes for this batch
    indexes = self.indexes[index * self.batch_size : (index + 1)* self.batch_size]

    # Gets list of names of input images for this batch
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Gets list of names of output images for this batch
    seg_IDs_temp = [self.seg_IDs[k] for k in indexes]

    X, y = self.__data_generation(list_IDs_temp, seg_IDs_temp)

    return X, y


class WordDataGenerator(LineDataGenerator):
  """
  Class that generates batches of the word segmented data
  """

  def __data_generation(self, list_IDs_temp, seg_IDs_temp):
    """
    Generates data containing batch_size samples
    This code is multi-core friendly
    """
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size, *self.dim, self.n_channels))

    for i, ID in enumerate(list_IDs_temp):
      img = cv2.imread(f'{self.in_folder}{ID}.jpg',0)
      img = utils.pad_img(img)
      img = utils.img_preprocessing(img)
      X[i,] = img
    
    for i, seg_ID in enumerate(seg_IDs_temp):
      seg = cv2.imread(f'{self.out_folder}{seg_ID}.png',0)
      seg = utils.pad_segment(seg)
      seg = cv2.resize(seg,(512,512))
      seg = np.stack((seg,)*3, axis=-1)
      seg = utils.get_seg_img(seg)
      y[i,] = seg

    return X, y
  
  def __getitem__(self, index):
    """
    Generates one batch of data
    """

    # Gets images indexes for this batch
    indexes = self.indexes[index * self.batch_size : (index + 1)* self.batch_size]

    # Gets list of names of input images for this batch
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Gets list of names of output images for this batch
    seg_IDs_temp = [self.seg_IDs[k] for k in indexes]

    X, y = self.__data_generation(list_IDs_temp, seg_IDs_temp)

    return X, y

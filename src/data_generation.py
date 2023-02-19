import numpy as np
# from google.colab.patches import cv2_imshow
import tensorflow as tf
import utils


class DataGenerator(tf.keras.utils.Sequence):
  """
  Class that generates batches of data
  Returns a batch of images and corresponding YOLO output label matrices
  """
  
  def __init__(self, in_folder, out_folder, list_IDs, label_IDs, batch_size = 4):
      self.in_folder = in_folder
      self.out_folder = out_folder
      self.list_IDs = list_IDs
      self.label_IDs = label_IDs
      self.batch_size = batch_size
      # self.on_epoch_end()

  def on_epoch_end(self):
    """
    Updates indexes after each ephoc
    """
    self.indexes = np.arange(len(self.list_IDs))

  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __data_generation(self, batch_x, batch_y):
    """
    Generates data containing batch_size samples
    This code is multi-core friendly
    """

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = self.in_folder + '/' + batch_x[i]
      label_path = self.out_folder + '/' + batch_y[i]

      image, label_matrix = utils.read(img_path, label_path)

      train_image.append(image)
      train_label.append(label_matrix)

    return np.array(train_image), np.array(train_label)

  def __getitem__(self, index):
    """
    Generates one batch of data
    """

    # Gets images indexes for this batch
    # indexes = self.indexes[index * self.batch_size : (index + 1)* self.batch_size]

    batch_x = self.list_IDs[index * self.batch_size : (index+1) * self.batch_size]
    batch_y = self.label_IDs[index * self.batch_size : (index+1) * self.batch_size]

    X, y = self.__data_generation(batch_x, batch_y)

    return X, y

import numpy as np
# from google.colab.patches import cv2_imshow
import tensorflow as tf
import cv2
import utils
import data_augmentation

import imgaug.augmenters as iaa
import pyclipper
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

mean = [103.939, 116.779, 123.68] # TODO


# class DataGenerator(tf.keras.utils.Sequence):
#   """
#   Class that generates batches of data
#   Returns a batch of images and corresponding YOLO output label matrices
#   """
  
#   def __init__(self, in_folder, out_folder, list_IDs, label_IDs, batch_size = 4):
#       self.in_folder = in_folder
#       self.out_folder = out_folder
#       self.list_IDs = list_IDs
#       self.label_IDs = label_IDs
#       self.batch_size = batch_size
#       # self.on_epoch_end()

#   def on_epoch_end(self):
#     """
#     Updates indexes after each ephoc
#     """
#     self.indexes = np.arange(len(self.list_IDs))

#   def __len__(self):
#     return int(np.floor(len(self.list_IDs) / self.batch_size))

#   def __data_generation(self, batch_x, batch_y):
#     """
#     Generates data containing batch_size samples
#     This code is multi-core friendly
#     """

#     train_image = []
#     train_label = []

#     for i in range(0, len(batch_x)):
#       img_path = self.in_folder + '/' + batch_x[i]
#       label_path = self.out_folder + '/' + batch_y[i]

#       image, label_matrix = utils.read(img_path, label_path)

#       train_image.append(image)
#       train_label.append(label_matrix)

#     return np.array(train_image), np.array(train_label)

#   def __getitem__(self, index):
#     """
#     Generates one batch of data
#     """

#     # Gets images indexes for this batch
#     # indexes = self.indexes[index * self.batch_size : (index + 1)* self.batch_size]

#     batch_x = self.list_IDs[index * self.batch_size : (index+1) * self.batch_size]
#     batch_y = self.label_IDs[index * self.batch_size : (index+1) * self.batch_size]

#     X, y = self.__data_generation(batch_x, batch_y)

#     return X, y



# binarization is the process of converting a grayscale or color image into a binary image,
# where each pixel is either black or white (or 0 or 1 / beloging to the class or not)
# pixels with values above the threshold are considered part of the object of interest,
# while those below the threshold are considered part of the background

# Differentiable Binarization (DB), which can perform the binarization process in a segmentation network
#  A segmentation network can adaptively set the thresholds for binarization, which not only
# simplifies the post-processing but also enhances the performance of text detection

# The network has feature pyramid backbone, and then a upsampling module to upsample the feature map into F
# F is used to predict the probability map (P) and the threshold map (T)
# After that, the binary map (B) is calculated by P and F

# The loss: Lp + alpha * Lb + beta * Lt
# Lp = Lb = cross entropy lodd
# Lt = sum of L1 distances between prediction and label inside the text polygon

# In the inference period, the bounding boxes can be obtained easily from the approximate binary map or the
# probability map by a box formulation module.

def compute_distance(xs, ys, point_1, point_2):
  """
    Computes the euclidean distance between a pixel and a line segment in the polygon
  """
  square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
  square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
  square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

  cosin = (square_distance - square_distance_1 - square_distance_2) / \
          (2 * np.sqrt(square_distance_1 * square_distance_2))
  square_sin = 1 - np.square(cosin)
  square_sin = np.nan_to_num(square_sin)

  if np.amin(square_distance_1 * square_distance_2 * square_sin / square_distance) < 0:
    print('Warning: value is less than 0') # TODO: ver se isto da problemas
  result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

  result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
  return result

def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
  """
    Draws the 2D threshold map in an image.
    Each pixel is assigned a value between 0 and 1, indicating its proximity to
    the polygon boundary. The closer, the higher the value.
  """
  polygon = np.array(polygon)
  assert polygon.ndim == 2
  assert polygon.shape[1] == 2

  # converts 2D array into a Polygon type
  polygon_shape = Polygon(polygon)
  distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
  subject = [tuple(l) for l in polygon]
  padding = pyclipper.PyclipperOffset()
  padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
  # calculate the distance between the original polygon and a padded polygon
  # the padded polygon is used to calculate the threshold values for each pixel
  # in the threshold map
  padded_polygon = np.array(padding.Execute(distance)[0])
  cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

  xmin = padded_polygon[:, 0].min()
  xmax = padded_polygon[:, 0].max()
  ymin = padded_polygon[:, 1].min()
  ymax = padded_polygon[:, 1].max()
  width = xmax - xmin + 1
  height = ymax - ymin + 1

  polygon[:, 0] = polygon[:, 0] - xmin
  polygon[:, 1] = polygon[:, 1] - ymin

  xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
  ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

  distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
  for i in range(polygon.shape[0]):
    j = (i + 1) % polygon.shape[0]
    # minimum distance between each pixel in the image and the polygon
    # for each line segment of the polygon
    absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
    # divided by the padded polygon distance and clipped into a range [0, 1]
    distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
  # the min value is taken to produce the threshold map
  distance_map = np.min(distance_map, axis=0)

  xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
  xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
  ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
  ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

  # print(xmax_valid)
  # print(xmin_valid)
  # print(ymin_valid)
  # print(ymax_valid)

  # print(distance_map.shape)
  # print(canvas.shape)

  # print(ymin_valid - ymin, ymax_valid - ymin)
  # print(xmin_valid - xmin, xmax_valid - xmin)
  # print('------')
  # print(ymin_valid, ymax_valid)
  # print(xmin_valid, xmax_valid)

  # sets the values in the canvas to thr max between the threshold map
  # and the existing canvas values, but only for the regions that overlap
  # with the polygon
  canvas[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(1 - distance_map[
                  ymin_valid - ymin:ymax_valid - ymin, xmin_valid - xmin:xmax_valid - xmin
                                                                                          ],
      canvas[ymin_valid:ymax_valid, xmin_valid:xmax_valid])

  return polygon, canvas, mask

def parse_annot(src_path):
  """
    Input: path to the annotation file
    Output: list of dictionaries containing the bounding box coordinates and the label for each word in image
  """
  annot = [] # list of dictionaries {'poly': , 'text' : }
  reader = open(src_path, 'r').readlines()
  # read a line
  for line in reader:

    word = {} # one dict per bounding box
    parts = line.strip().split(',') # '641', '173', '656', '168', '657', '181', '643', '187', '###']
    label = parts[-1]

    # edge case
    if label == '1':
        label = '###'

    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
    # extract the polygon coordinates: 2D array with x and y coordinates
    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist() # [[641.0, 173.0], [656.0, 168.0], [657.0, 181.0], [643.0, 187.0]]
    if len(poly) < 3:
        continue

    word['poly'] = poly
    word['text'] = label
    annot.append(word)

  return annot


class DataGenerator(tf.keras.utils.Sequence):
  """
  Class that generates batches of data
  Returns a batch of images and corresponding YOLO output label matrices
  """
  
  # list_IDs = ['img1.jpg', 'img2.jpg', etc]
  # label_IDs = ['img1.txt', 'img2.txt', etc]
  
  def __init__(self, in_folder, label_folder, list_IDs, label_IDs, batch_size = 16,
               img_size = 640, min_text_size = 8, shrink_ratio = 0.4, thresh_min = 0.3, thresh_max = 0.7, training = True):
      self.in_folder = in_folder
      self.label_folder = label_folder
      self.list_IDs = list_IDs
      self.label_IDs = label_IDs
      self.batch_size = batch_size
      self.img_size = img_size
      self.min_text_size = min_text_size
      self.shrink_ratio = shrink_ratio
      self.thresh_min = thresh_min
      self.thresh_max = thresh_max
      self.training = training
      # self.on_epoch_end()

  def on_epoch_end(self):
    """
    Updates indexes after each ephoc
    """
    self.indexes = np.arange(len(self.list_IDs))

  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def preprocess_img_annot(self, image, annots):

  # augmentation:
  # - random rotation with an angle of (-10º, 10º)
  # - random cropping
  # - random flipping
    transform_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10)), iaa.Resize((0.5, 3.0))])

    if self.training:
      transform_aug = transform_aug.to_deterministic()
      image, annots = data_augmentation.transform(transform_aug, image, annots)
      image, annots = data_augmentation.crop(image, annots)

    image, annots = utils.resize(self.img_size, image, annots)
    
    annots = [ann for ann in annots if Polygon(ann['poly']).is_valid]

    return image, annots

  def generate_maps(self, anns):
    # initialize maps
    gt = np.zeros((self.img_size, self.img_size), dtype=np.float32)
    mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
    thresh_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
    thresh_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

    # for each annotation in image
    for ann in anns:
      # get annotation polygon
      poly = np.array(ann['poly']) # [[641.0, 173.0], [656.0, 168.0], [657.0, 181.0], [643.0, 187.0]]
      # height and width
      height = max(poly[:, 1]) - min(poly[:, 1])
      width = max(poly[:, 0]) - min(poly[:, 0])
      # polygon
      polygon = Polygon(poly)
      # generate gt and mask
      if polygon.area < 1 or min(height, width) < self.min_text_size or ann['text'] == '###':
        cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        continue
      else:
        distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
        subject = [tuple(l) for l in ann['poly']]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)
        if len(shrinked) == 0:
          cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
          continue
        else:
          shrinked = np.array(shrinked[0]).reshape(-1, 2)
          if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
            cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
          else:
            cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            continue
      # generate thresh map and thresh mask
      # print(thresh_mask.shape)
      # print(thresh_map.shape)
      ann['poly'], thresh_map, thresh_mask = draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio = self.shrink_ratio)

    thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

    return gt, mask, thresh_map, thresh_mask

  def __data_generation(self, batch_x, batch_y):
    """
    Generates data containing batch_size samples
    This code is multi-core friendly
    """

    images = []
    bboxes = []
    masks = []
    thresh_maps = []
    thresh_masks = []
    batch_loss = np.zeros([len(batch_x), ], dtype=np.float32)
    
    # all annotations from that paths
    # all_anns = load_all_anns(gt_paths)


    # batch_images = np.zeros([batch_size, image_size, image_size, 3], dtype=np.float32) -> images
    # batch_gts = np.zeros([batch_size, image_size, image_size], dtype=np.float32) -> ?
    # batch_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32) -> ?
    # batch_thresh_maps = np.zeros([batch_size, image_size, image_size], dtype=np.float32) -> ?
    # batch_thresh_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32) -> ?
    # batch_loss = np.zeros([batch_size, ], dtype=np.float32)

    for i in range(0, len(batch_x)):
      # image path and label path
      img_path = self.in_folder + '/' + batch_x[i] # '/content/drive/My Drive/Colab Notebooks/ICDAR2015/Challenge4/ch4_training_images/img_1.jpg'
      annot_path = self.label_folder + '/' + batch_y[i] # '/content/drive/My Drive/Colab Notebooks/ICDAR2015/Challenge4/ch4_training_localization_transcription_gt/gt_img_1.txt'

      # read image image
      # print(img_path)
      image = cv2.imread(img_path)

      # plt.imshow(image)
      # plt.show()
      # # specific image annotations
      anns = parse_annot(annot_path) # get image annotations [{'poly', 'text'}, {'poly', 'text'}, etc.}]
      # print(anns)
      # data augmentation if training + image resizing
      image, anns = self.preprocess_img_annot(image, anns)

      # plt.imshow(image)
      # plt.show()

      # get different types of annotations
      gt, mask, thresh_map, thresh_mask = self.generate_maps(anns)

      # image scaling
      image = image.astype(np.float32)
      image[..., 0] -= mean[0]
      image[..., 1] -= mean[1]
      image[..., 2] -= mean[2]

      # plt.imshow(image)
      # plt.show()

      images.append(image)
      bboxes.append(gt)
      masks.append(mask)
      thresh_maps.append(thresh_map)
      thresh_masks.append(thresh_mask)

    inputs = [np.array(images), np.array(bboxes), np.array(masks), np.array(thresh_maps), np.array(thresh_masks)]
    outputs = batch_loss
    
    return inputs, outputs

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
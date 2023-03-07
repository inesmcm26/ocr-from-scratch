import tensorflow as tf
import data_generation
import keras.backend as K
# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

LAMBDA_COORDS = 5
LAMBDA_NOOBJ = 0.5


def train_model(model, img_path, labels_path, images, labels, batch_size, epochs = 30):

  training_generator = data_generation.DataGenerator(img_path, labels_path, images['train'], labels['train'], batch_size)

  validation_generator = data_generation.DataGenerator(img_path, labels_path, images['train'], labels['train'], batch_size)

  opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

  model.compile(loss=yolo_loss , optimizer = opt)

  mcp_save = ModelCheckpoint('/home/inesmcm/Desktop/projects/ocr-from-scratch/models/yolo_models/yolo_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1,)

  history = model.fit(x = training_generator,
          steps_per_epoch = int(len(images['train']) // batch_size),
          epochs = epochs,
          verbose = 1,
          workers= 4,
          validation_data = validation_generator,
          validation_steps = int(len(images['validation']) // batch_size),
          callbacks=[
              mcp_save
          ])

  return history, model


# class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
#   """

#   The original paper uses a customized leraning rate for different epochs.

#   Input: schedule, a function that takes an epoch index and current learning rate and
#   returns a new learning rate as output.

#   """

#   def __init__(self, schedule):
#       super(CustomLearningRateScheduler, self).__init__()
#       self.schedule = schedule

#   def on_epoch_begin(self, epoch, logs=None):
#       # Check if optimizer has a learning rate attribute

#       if not hasattr(self.model.optimizer, "lr"):
#           raise ValueError('Optimizer must have a "lr" attribute.')
      
#       # Get the current learning rate
#       lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

#       # Get the new learning rate
#       scheduled_lr = self.schedule(epoch, lr)

#       # Set the new learning rate on the optimizer
#       tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

#       print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


# LR_SCHEDULE = [
#     (0, 0.01),
#     (75, 0.001),
#     (105, 0.0001),
# ]


# def lr_schedule(epoch, lr):
#     """
#     Auxiliary function that returns the learning rate shcedule for a specific epoch
#     """
#     # if epoch is before the first turning point, return the initial learning rate
#     # if epoch is after the last turning point, return the final learning rate
#     if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
#         return lr
    
#     # if epoch is a turning point, return the corresponding learning rate
#     for i in range(len(LR_SCHEDULE)):
#         if epoch == LR_SCHEDULE[i][0]:
#             return LR_SCHEDULE[i][1]
        
#     # else return the same learning rate
#     return lr

# def xywh2minmax(xy, wh):
#   """
#   Converts the bounding box format from (x, y, w, h) to (xmin, ymin, xmax, ymax)
#   """
#   xy_min = xy - wh / 2
#   xy_max = xy + wh / 2

#   return xy_min, xy_max


# def iou(pred_mins, pred_maxes, true_mins, true_maxes):
#   """
#   Calculates the Intersection over Union (IoU) between two bounding boxes.
#   """

#   # Calculate the intersection areas
#   intersect_mins = K.maximum(pred_mins, true_mins)
#   intersect_maxes = K.minimum(pred_maxes, true_maxes)
#   intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
#   intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

#   # Calculate the union areas
#   pred_wh = pred_maxes - pred_mins
#   true_wh = true_maxes - true_mins
#   pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
#   true_areas = true_wh[..., 0] * true_wh[..., 1]

#   # Calculate the IoU
#   union_areas = pred_areas + true_areas - intersect_areas
#   iou_scores = intersect_areas / union_areas

#   return iou_scores


# def yolo_head(feats):
#   """
#   Converts the final layer features to bounding box parameters.
#   """

#   # Dynamic implementation of conv dims for fully convolutional model.
#   conv_dims = K.shape(feats)[1:3]  # assuming channels last

#   # In YOLO the height index is the inner most iteration.
#   conv_height_index = K.arange(0, stop=conv_dims[0])
#   conv_width_index = K.arange(0, stop=conv_dims[1])
#   conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

#   # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
#   # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
#   conv_width_index = K.tile(
#       K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
#   conv_width_index = K.flatten(K.transpose(conv_width_index))
#   conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
#   conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
#   conv_index = K.cast(conv_index, K.dtype(feats))

#   conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

#   box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
#   box_wh = feats[..., 2:4] * 448

#   return box_xy, box_wh


# def yolo_loss(y_true, y_pred):
#   # y_true: (batch_size, grid, grid, [class, x, y, w, h, confidence])
#   label_class = y_true[..., :1]  # ? x 15 x 15 x 1 class for each grid cell
#   label_box = y_true[..., 1:5]  # ? x 15 x 15 x 4 x, y, w, h
#   response_mask = y_true[..., 5]  # ? x 15 x 15 confidence
#   response_mask = K.expand_dims(response_mask)  # ? x 15 x 15 x 1

#   # y_pred: (batch_size, grid, grid, [x, y, w, h, confidence, class])
#   predict_class = y_pred[..., :1]  # ? x 15 x 15 x 1 class for each grid cell
#   predict_confidence = y_pred[..., 1:3]  # ? x 15 x 15 x 2 -> confidence
#   predict_box = y_pred[..., 3:]  # ? x 15 x 15 x 8 x, y, w and h for each anchor

#   _label_box = K.reshape(label_box, [-1, 15, 15, 1, 4])
#   _predict_box = K.reshape(predict_box, [-1, 15, 15, 2, 4])

#   # Convert label_box and predict_box to (x,y) and (w,h) format for easier calculations
#   label_xy, label_wh = yolo_head(_label_box)  # ? x 15 x 15 x 1 x 2, ? x 15 x 15 x 1 x 2
#   label_xy = K.expand_dims(label_xy, 3)  # ? x 15 x 15 x 1 x 1 x 2
#   label_wh = K.expand_dims(label_wh, 3)  # ? x 15 x 15 x 1 x 1 x 2
#   label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? x 15 x 15 x 1 x 1 x 2, ? x 15 x 15 x 1 x 1 x 2

#   predict_xy, predict_wh = yolo_head(_predict_box)  # ? x 15 x 15 x 2 x 2, ? x 15 x 15 x 2 x 2
#   predict_xy = K.expand_dims(predict_xy, 4)  # ? x 15 x 15 x 2 x 1 x 2
#   predict_wh = K.expand_dims(predict_wh, 4)  # ? x 15 x 15 x 2 x 1 x 2
#   predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? x 15 x 15 x 2 x 1 x 2, ? x 15 x 15 x 2 x 1 x 2

#   # Calculate intersection-over-union (IOU) scores between predicted and actual boxes
#   iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? x 15 x 15 x 2 x 1
#   best_ious = K.max(iou_scores, axis = 4)  # ? x 15 x 15 x 2
#   best_box = K.max(best_ious, axis=3, keepdims=True)  # ? x 15 x 15 x 1

#   # Calculate box_mask, which is used to identify the best anchor box for each grid cell
#   box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? x 15 x 15 x 2

#   # Calculate loss for confidence scores
#   no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_confidence)
#   object_loss = box_mask * response_mask * K.square(1 - predict_confidence)
#   confidence_loss = no_object_loss + object_loss
#   confidence_loss = K.sum(confidence_loss)

#   # Calculate loss for class labels
#   class_loss = response_mask * K.square(label_class - predict_class)
#   class_loss = K.sum(class_loss)

#   # Reshape label_box and predict_box for further calculations
#   _label_box = K.reshape(label_box, [-1, 15, 15, 1, 4])
#   _predict_box = K.reshape(predict_box, [-1, 15, 15, 2, 4])

#   # Convert label_box and predict_box to (x,y) and (w,h) format for easier calculations
#   label_xy, label_wh = yolo_head(_label_box)  # ? x 15 x 15 x 1 x 2, ? x 15 x 15 x 1 x 2
#   predict_xy, predict_wh = yolo_head(_predict_box)  # ? x 15 x 15 x 2 x 2, ? x 15 x 15 x 2 x 2

#   # Calculate loss for box coordinates
#   box_mask = K.expand_dims(box_mask)
#   response_mask = K.expand_dims(response_mask)

#   # Calculate loss for box coordinates
#   box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
#   box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
#   box_loss = K.sum(box_loss)

#   # Calculate total loss
#   loss = confidence_loss + class_loss + box_loss

#   return loss

# def yolo_loss(y_true, y_pred, S=15, B=2, C=1, lambda_coord=5, lambda_noobj=0.5):
#   # Reshape y_true and y_pred
#   y_true = tf.reshape(y_true, (-1, S, S, C + 5 * B))
#   y_pred = tf.reshape(y_pred, (-1, S, S, C + 5 * B))

#   # Split y_true and y_pred into their respective components
#   pred_box_xy = y_pred[:, :, :, :2]
#   pred_box_wh = y_pred[:, :, :, 2:4]
#   pred_box_confidence = y_pred[:, :, :, 4:5]
#   pred_box_class_probs = y_pred[:, :, :, 5:]

#   true_box_xy = y_true[:, :, :, :2]
#   true_box_wh = y_true[:, :, :, 2:4]
#   true_box_confidence = y_true[:, :, :, 4:5]
#   true_box_class_probs = y_true[:, :, :, 5:]

#   # Calculate the coordinates of the boxes' top-left corners and bottom-right corners
#   pred_box_mins = pred_box_xy - 0.5 * pred_box_wh
#   pred_box_maxes = pred_box_xy + 0.5 * pred_box_wh
#   true_box_mins = true_box_xy - 0.5 * true_box_wh
#   true_box_maxes = true_box_xy + 0.5 * true_box_wh

#   # Calculate the intersection of the boxes
#   intersect_mins = tf.maximum(pred_box_mins, true_box_mins)
#   intersect_maxes = tf.minimum(pred_box_maxes, true_box_maxes)
#   intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0)
#   intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

#   # Calculate the union of the boxes
#   pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
#   true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
#   union_areas = pred_areas + true_areas - intersect_areas

#   # Calculate the intersection over union (iou)
#   iou_scores = intersect_areas / union_areas

#   # Calculate the confidence loss
#   pred_box_obj = pred_box_confidence * tf.expand_dims(tf.cast(iou_scores >= 0.5, dtype=y_pred.dtype), axis=-1)
#   true_box_obj = true_box_confidence
#   confidence_loss = tf.reduce_sum(tf.square(true_box_obj - pred_box_obj), axis=[1, 2, 3])

#   # Calculate the localization loss
#   pred_box_xy_normalized = pred_box_xy / tf.constant([S, S], dtype=y_pred.dtype)
#   pred_box_wh_normalized = pred_box_wh / tf.constant([S, S], dtype=y_pred.dtype)
#   true_box_xy_normalized = true_box_xy / tf.constant([S, S], dtype=y_pred.dtype)
#   true_box_wh_normalized = true_box_wh / tf.constant([S, S], dtype=y_pred.dtype)
#   localization_loss = tf.reduce_sum(tf.square(true_box_xy_normalized - pred_box_xy_normalized), axis=[1, 2, 3]) + tf.reduce_sum(tf.square(tf.sqrt(true_box_wh_normalized) - tf.sqrt(pred_box_wh_normalized)), axis=[1, 2, 3])

#   # Calculate the class loss
#   class_loss = tf.reduce_sum(tf.square(true_box_class_probs - pred_box_class_probs), axis=[1, 2, 3])

# 
#   object_mask = tf.squeeze(true_box_confidence, axis=-1)
#   ignore_mask = tf.cast(iou_scores < 0.5, dtype=object_mask.dtype)
#   object_loss = lambda_coord * tf.reduce_sum(object_mask * (localization_loss + lambda_noobj * (1 - object_mask) * confidence_loss))
#   no_object_loss = lambda_noobj * tf.reduce_sum(ignore_mask * confidence_loss)

#   yolo_loss = object_loss + no_object_loss + tf.reduce_sum(class_loss)

#   return yolo_loss

def yolo_loss(y_true, y_pred):
  # y_true = [batch_size, GRID_W, GRID_H, 1, 5]
  # y_true[4] = [confidence, x, y, w, h]
  coords = y_true[:, :, :, :, 0] * LAMBDA_COORDS
  noobj = (-1 * (y_true[:, :, :, :, 0] - 1) * LAMBDA_NOOBJ)

  pc_true = y_true[:, :, :, :, 0]
  pc_pred = y_pred[:, :, :, :, 0]

  x_true = y_true[:, :, :, :, 1]
  x_pred = y_pred[:, :, :, :, 1]

  yy_true = y_true[:, :, :, :, 2]
  yy_pred = y_pred[:, :, :, :, 2]

  w_true = y_true[:, :, :, :, 3]
  w_pred = y_pred[:, :, :, :, 3]

  h_true = y_true[:, :, :, :, 4]
  h_pred = y_pred[:, :, :, :, 4]

  p_loss_absent = K.sum(K.square(pc_pred - pc_true) * noobj)
  p_loss_present = K.sum(K.square(pc_pred - pc_true))

  x_loss = K.sum(K.square(x_pred - x_true) * coords)
  yy_loss = K.sum(K.square(yy_pred - yy_true) * coords)

  w_loss = K.sum(K.square(K.sqrt(w_pred) - K.sqrt(w_true)) * coords)
  h_loss = K.sum(K.square(K.sqrt(h_pred) - K.sqrt(h_true)) * coords)

  loss = p_loss_absent + p_loss_present + x_loss + yy_loss + w_loss + h_loss

  return loss
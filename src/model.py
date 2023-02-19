import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2

class Yolo_Reshape(tf.keras.layers.Layer):
  """
  Reshapes the output of the YOLO model to the required output shape
  """

  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 15x15
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 1
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B
    
    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs
  

def yolov1(input_shape = (418, 418, 3)):

    lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

    inpt = Input(input_shape)
    conv1 = Conv2D(filters=64, kernel_size= 7, strides= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(inpt)
    max1 = MaxPooling2D(pool_size= 2, strides = 2, padding = 'same')(conv1)

    conv2 = Conv2D(filters=192, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(max1)
    max2 = MaxPooling2D(pool_size= 2, strides = 2, padding = 'same')(conv2)

    conv3 = Conv2D(filters=128, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(max2)
    conv3 = Conv2D(filters=256, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv3)
    conv3 = Conv2D(filters=256, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv3)
    conv3 = Conv2D(filters=512, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv3)
    max3 = MaxPooling2D(pool_size= 2, strides = 2, padding = 'same')(conv3)
    
    conv4 = Conv2D(filters=256, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(max3)
    conv4 = Conv2D(filters=512, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=256, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=512, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=256, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=512, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=256, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=512, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=512, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    conv4 = Conv2D(filters=1024, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv4)
    max4 = MaxPooling2D(pool_size= 2, strides = 2, padding = 'same')(conv4)

    conv5 = Conv2D(filters=512, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(max4)
    conv5 = Conv2D(filters=1024, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv5)
    conv5 = Conv2D(filters=512, kernel_size= 1, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv5)
    conv5 = Conv2D(filters=1024, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv5)
    conv5 = Conv2D(filters=1024, kernel_size= 3, padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4))(conv5)
    conv5 = Conv2D(filters=1024, kernel_size= 3, strides= 2, padding = 'same')(conv5)

    conv6 = Conv2D(filters=1024, kernel_size= 3, activation=lrelu, kernel_regularizer=l2(5e-4))(conv5)
    conv6 = Conv2D(filters=1024, kernel_size= 3, activation=lrelu, kernel_regularizer=l2(5e-4))(conv6)

    flat = Flatten()(conv6)
    dense1 = Dense(512)(flat)
    dense2 = Dense(1024)(dense1)
    drop = Dropout(0.5)(dense2)
    dense3 = Dense(1470, activation = 'sigmoid')(drop)
    out = Yolo_Reshape(target_shape=(15,15,11))(dense3)

    model = Model(inpt, out)

    return model

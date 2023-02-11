import os
import tensorflow as tf
import utils
import data_generation


def train_model(model, img_path, seg_path, images, seg_images):

  training_generator = data_generation.DataGenerator(img_path, seg_path, images['train'], seg_images['train'])
  validation_generator = data_generation.DataGenerator(img_path, seg_path, images['validation'], seg_images['validation'])

  checkpoint_path = "/content/drive/MyDrive/Colab Notebooks/OCR/models/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


  model.fit(training_generator,
          validation_data=validation_generator,
          epochs=3,
          use_multiprocessing=True,
          workers=6,
          callbacks=[cp_callback])

  return model
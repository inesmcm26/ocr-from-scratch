import os
import tensorflow as tf
import utils
import data_generation


def train_model(model, img_path, seg_path, images, seg_images, task, batch_size, epochs):

  if task == 'line_segmentation':
    training_generator = data_generation.LineDataGenerator(img_path, seg_path, images['train'], seg_images['train'])
    validation_generator = data_generation.LineDataGenerator(img_path, seg_path, images['validation'], seg_images['validation'])
  else:
    training_generator = data_generation.WordDataGenerator(img_path, seg_path, images['train'], seg_images['train'])
    validation_generator = data_generation.WordDataGenerator(img_path, seg_path, images['validation'], seg_images['validation'])

  # checkpoint_path = "/content/drive/MyDrive/Colab Notebooks/OCR/models/cp.ckpt"
  # checkpoint_dir = os.path.dirname(checkpoint_path)

  # # Create a callback that saves the model's weights
  # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
  #                                                save_weights_only=True,
  #                                                verbose=1)


  model.fit(training_generator,
          validation_data = validation_generator,
          epochs = epochs,
          use_multiprocessing = True,
          workers = 6,
          epochs=5,steps_per_epoch=1000,
          validation_steps=400)
          # callbacks=[cp_callback])

  return model
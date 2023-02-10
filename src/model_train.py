import utils

from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, X_train, X_test, y_train, y_test, img_path, seg_path):
  mc = ModelCheckpoint('weights{epoch:08d}.h5', 
                                      save_weights_only=True, period=1)
  model.fit_generator(utils.batch_generator(X_train, y_train, img_path, seg_path, 2), epochs=3, steps_per_epoch=1000,
                      validation_data=utils.batch_generator(X_test, y_test, img_path, seg_path, 2),
                      validation_steps=400, callbacks=[mc], shuffle = 1, verbose = True)
  return model
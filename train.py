import tensorflow as tf
import keras
from reg_cnn import get_model
from datagen import DataGenerator
from matplotlib import pyplot as plt
import os
import numpy as np
np.set_printoptions(precision=2)
import config as cfg

path = os.path.dirname(os.path.abspath(__file__))
train_gen = DataGenerator(path='cat_dog/cats_and_dogs_filtered/train')
val_gen = DataGenerator(path='cat_dog/cats_and_dogs_filtered/validation')


if 1:
    model = tf.keras.models.load_model(cfg.checkpoint_path)
else:
    input_shape=(128,128,3)
    model = get_model(input_shape)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,loss = 'mse',metrics =[tf.keras.metrics.MeanSquaredError()])

model.summary()
checkpoint = keras.callbacks.ModelCheckpoint(cfg.checkpoint_path,monitor='val_mean_squared_error',verbose=1,save_best_only=True,save_weights_only=False)

callbacks = [checkpoint]
if 1:
    history = model.fit(train_gen, validation_data=val_gen, epochs=100,callbacks=callbacks)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#model.evaluate(val_gen)
#import pdb;pdb.set_trace()

ypred = model.predict(val_gen)

for i in range(3):
    batch_x, batch_y  = val_gen.__getitem__(0)
    pred_y = model.predict(batch_x)
    for i in range(len(pred_y)):
        print(f'actual:{batch_y[i]}, pred: {pred_y[i][0]}')

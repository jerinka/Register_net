from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import tensorflow as tf

data, target = load_iris(return_X_y=True)
X = data[:, (0, 1, 2)]
Y = data[:, 3]
Z = target

inputs = Input(shape=(3,), name='input')
x = Dense(16, activation='relu', name='16')(inputs)
x = Dense(32, activation='relu', name='32')(x)
output1 = Dense(1, name='cont_out')(x)
output2 = Dense(3, activation='softmax', name='cat_out')(x)

model = Model(inputs=inputs, outputs=[output1, output2])

model.compile(loss={'cont_out': 'mean_absolute_error', 
                    'cat_out': 'sparse_categorical_crossentropy'},
              optimizer='adam',
              metrics={'cat_out': tf.metrics.SparseCategoricalAccuracy(name='acc')})

model.summary()

history = model.fit(X, {'cont_out': Y, 'cat_out': Z}, epochs=10, batch_size=8)

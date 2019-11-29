# -*- coding: utf-8 -*-
"""
Training models on images after preprocessing 1
"""

from __future__ import print_function
import tensorflow as tf

import pickle

import numpy as np


import os
DATASET_PATH = './'
TRAIN_PATH_NORMAL = os.path.join(DATASET_PATH, 'normal_train.pkl')
TRAIN_PATH_PNEUMONIA = os.path.join(DATASET_PATH, 'pneumonia_train.pkl')
VALIDATION_PATH_NORMAL = os.path.join(DATASET_PATH, 'normal_val.pkl')
VALIDATION_PATH_PNEUMONIA = os.path.join(DATASET_PATH, 'pneumonia_val.pkl')
TEST_PATH_NORMAL = os.path.join(DATASET_PATH, 'normal_test.pkl')
TEST_PATH_PNEUMONIA = os.path.join(DATASET_PATH, 'pneumonia_test.pkl')

file = open(TRAIN_PATH_NORMAL, 'rb')
# dump information to that file
train_normal = pickle.load(file)
# close the file
file.close()

file = open(TRAIN_PATH_PNEUMONIA, 'rb')
# dump information to that file
train_pneumonia = pickle.load(file)
# close the file
file.close()

file = open(VALIDATION_PATH_NORMAL, 'rb')
# dump information to that file
val_normal = pickle.load(file)
# close the file
file.close()

file = open(VALIDATION_PATH_PNEUMONIA, 'rb')
# dump information to that file
val_pneumonia = pickle.load(file)
# close the file
file.close()

file = open(TEST_PATH_NORMAL, 'rb')
# dump information to that file
test_normal = pickle.load(file)
# close the file
file.close()

file = open(TEST_PATH_PNEUMONIA, 'rb')
# dump information to that file
test_pneumonia = pickle.load(file)
# close the file
file.close()

y_normal = np.zeros((1341,), dtype=int)
y_pneumonia = np.ones((3882,), dtype=int)

x_train = np.concatenate((train_normal, train_pneumonia))
y_train = np.concatenate((y_normal, y_pneumonia))

y_normal_t = np.zeros((234,), dtype=int)
y_pneumonia_t = np.ones((390,), dtype=int)

x_test = np.concatenate((test_normal, test_pneumonia))
y_test = np.concatenate((y_normal_t, y_pneumonia_t))


from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)


# input image dimensions
img_rows, img_cols = 150, 125

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 2
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 input_shape=(150, 125, 1)),
          tf.keras.layers.MaxPooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.MaxPooling2D(2, 1),
          tf.keras.layers.Conv2D(64, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.MaxPooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.MaxPooling2D(2, 1),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(2, activation='softmax')
  ])

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 input_shape=(150, 125, 1)),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(64, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(2, activation='softmax')
  ])

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 input_shape=(150, 125, 1)),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(64, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(256, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(2, activation='softmax')
  ])

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), input_shape=(150, 125, 1), padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

batch_size = 32
epochs = 40

from keras.callbacks import ModelCheckpoint, EarlyStopping
mcp = ModelCheckpoint(filepath='./model_simple.h5',monitor="val_acc", save_best_only=True, save_weights_only=False)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[mcp],
          validation_split=0.4)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# model.save('./model_inc1.h5')

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('w')
ax.grid(b=False)
ax.plot(hist.history['acc'], color='red')
ax.plot(hist.history['val_acc'], color ='green')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('w')
ax.grid(b=False)
ax.plot(hist.history['loss'], color='red')
ax.plot(hist.history['val_loss'], color ='green')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

tf.keras.utils.plot_model(model, to_file='./ave_model.png', show_shapes=True)


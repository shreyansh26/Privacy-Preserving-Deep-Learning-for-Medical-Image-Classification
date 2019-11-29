# -*- coding: utf-8 -*-
"""
Training using Differential Privacy
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
print('y_train shape:', y_train.shape)

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

dpsgd = True            # If True, train with DP-SGD
learning_rate = 0.015    # Learning rate for training
noise_multiplier = 1.1  # Ratio of the standard deviation to the clipping norm
l2_norm_clip = 1.0      # Clipping norm
batch_size = 50        # Batch size
epochs = 20             # Number of epochs
microbatches = 5       # Number of microbatches


def compute_epsilon(steps):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / 60000
    rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multiplier,
                    steps=steps,
                    orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

tf.logging.set_verbosity(tf.logging.INFO)
if dpsgd and batch_size % microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

# model = tf.keras.Sequential([
#       tf.keras.layers.Conv2D(16, 8,
#                              strides=2,
#                              padding='same',
#                              activation='relu',
#                              input_shape=(150, 125, 1)),
#       tf.keras.layers.AveragePooling2D(2, 1),
#       tf.keras.layers.Conv2D(32, 4,
#                              strides=2,
#                              padding='valid',
#                              activation='relu'),
#       tf.keras.layers.AveragePooling2D(2, 1),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(32, activation='relu'),
#       tf.keras.layers.Dense(2)
#   ])

# model = tf.keras.Sequential([
#           tf.keras.layers.Conv2D(16, 8,
#                                  strides=2,
#                                  padding='same',
#                                  activation='relu',
#                                  input_shape=(150, 125, 1)),
#           tf.keras.layers.MaxPooling2D(2, 1),
#           tf.keras.layers.Conv2D(32, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.MaxPooling2D(2, 1),
#           tf.keras.layers.Conv2D(64, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.MaxPooling2D(2, 1),
#           tf.keras.layers.Conv2D(32, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.MaxPooling2D(2, 1),
#           tf.keras.layers.Flatten(),
#           tf.keras.layers.Dense(32, activation='relu'),
#           tf.keras.layers.Dense(2)
#   ])

# model = tf.keras.Sequential([
#           tf.keras.layers.Conv2D(16, 8,
#                                  strides=2,
#                                  padding='same',
#                                  activation='relu',
#                                  input_shape=(150, 125, 1)),
#           tf.keras.layers.AveragePooling2D(2, 1),
#           tf.keras.layers.Conv2D(32, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.AveragePooling2D(2, 1),
#           tf.keras.layers.Conv2D(64, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.AveragePooling2D(2, 1),
#           tf.keras.layers.Conv2D(32, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.AveragePooling2D(2, 1),
#           tf.keras.layers.Flatten(),
#           tf.keras.layers.Dense(32, activation='relu'),
#           tf.keras.layers.Dense(2, activation='softmax')
#   ])

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

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

if dpsgd:
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches,
        learning_rate=learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
else:
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compile model with Keras
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

rows = x_train.shape[0]
rows_req = (rows//50)*50
x_train = x_train[:rows_req,:,:,:]
y_train = y_train[:rows_req,:]

x_train.shape

rows = x_test.shape[0]
rows_req = (rows//50)*50
x_test = x_test[:rows_req,:,:,:]
y_test = y_test[:rows_req,:]

x_test.shape

y_test[:30]

from keras.callbacks import ModelCheckpoint, EarlyStopping
mcp = ModelCheckpoint(filepath='./ave_model_dp.h5',monitor="val_acc", save_best_only=True, save_weights_only=False)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Compute the privacy budget expended.
if dpsgd:
    eps = compute_epsilon(epochs * 60000 // batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
else:
    print('Trained with vanilla non-private SGD optimizer')

# model.save('/content/drive/My Drive/BTP/models/simple_dp.h5')

x_test = x_test[:576,:,:,:]
y_test = y_test[:576,:]

# model.load_weights('/content/drive/My Drive/BTP/final/ave_model_dp.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# x_test[0].shape
# model.predict(x_test[0], y_test[0])

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
ax.plot(hist.history['loss'][10], color='red')
ax.plot(hist.history['val_loss'][10], color ='green')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


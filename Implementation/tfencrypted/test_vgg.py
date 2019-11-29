from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from collections import OrderedDict
import keras
import numpy as np

import tf_encrypted as tfe
import tf_encrypted.keras.backend as KE
from subprocess import call

input_shape = (224, 224, 3)
num_classes = 10
epochs = 2

# model = tf.keras.Sequential([
#           tf.keras.layers.Conv2D(16, 8,
#                                  strides=2,
#                                  padding='same',
#                                  activation='relu',
#                                  batch_input_shape=input_shape),
#           tf.keras.layers.AveragePooling2D(2, 1),
#           tf.keras.layers.Conv2D(32, 4,
#                                  strides=2,
#                                  padding='valid',
#                                  activation='relu'),
#           tf.keras.layers.AveragePooling2D(2, 1),
#           tf.keras.layers.Flatten(),
#           tf.keras.layers.Dense(32, activation='relu'),
#           tf.keras.layers.Dense(num_classes, name='logit')
#   ])

# pre_trained_weights = 'short-dnn.h5'
# model.load_weights(pre_trained_weights)
model = Sequential([
	Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
	Conv2D(64, (3, 3), activation='relu', padding='same'),
	MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Conv2D(128, (3, 3), activation='relu', padding='same'),
	Conv2D(128, (3, 3), activation='relu', padding='same',),
	MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Conv2D(256, (3, 3), activation='relu', padding='same',),
	Conv2D(256, (3, 3), activation='relu', padding='same',),
	Conv2D(256, (3, 3), activation='relu', padding='same',),
	MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Conv2D(512, (3, 3), activation='relu', padding='same',),
	Conv2D(512, (3, 3), activation='relu', padding='same',),
	Conv2D(512, (3, 3), activation='relu', padding='same',),
	MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Conv2D(512, (3, 3), activation='relu', padding='same',),
	Conv2D(512, (3, 3), activation='relu', padding='same',),
	Conv2D(512, (3, 3), activation='relu', padding='same',),
	MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Flatten(),
	Dense(4096, activation='relu'),
	Dense(4096, activation='relu'),
	Dense(1000, activation='softmax')
])

# model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=1000)

# model.layers.pop(0)
# pre_trained_weights = 'vgg.h5'
# model.load_weights(pre_trained_weights)

print(model.summary())

pre_trained_weights = 'vgg.h5'
model.load_weights(pre_trained_weights)

players = OrderedDict([
    ('server0', 'localhost:4000'),
    ('server1', 'localhost:4001'),
    ('server2', 'localhost:4002'),
])

config = tfe.RemoteConfig(players)
config.save('/tmp/tfe.config')

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

# for player_name in players.keys():
#     call("python3.6 -m tf_encrypted.player --config /tmp/tfe.config {}".format(player_name), shell=True)


tf.reset_default_graph()
with tfe.protocol.SecureNN():
    tfe_model = tfe.keras.models.clone_model(model)

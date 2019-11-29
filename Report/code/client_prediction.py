# coding: utf-8
# Private Prediction using TFE Keras - Serving (Client)
# 

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tensorflow.keras.datasets import mnist


# ## Data

import pickle
file = open('../normal_test.pkl', 'rb')
# dump information to that file
test_normal = pickle.load(file)
# close the file
file.close()

file = open('../pneumonia_test.pkl', 'rb')
# dump information to that file
test_pneumonia = pickle.load(file)
# close the file
file.close()


y_normal_t = np.zeros((234,), dtype=int)
y_pneumonia_t = np.ones((390,), dtype=int)

x_test = np.concatenate((test_normal, test_pneumonia))
y_test = np.concatenate((y_normal_t, y_pneumonia_t))
from sklearn.utils import shuffle
x_test, y_test = shuffle(x_test, y_test, random_state=0)
x_test.shape


rows = x_test.shape[0]
rows_req = (rows//50)*50
x_test = x_test[:rows_req,:,:]
y_test = y_test[:rows_req]


# input image dimensions
img_rows, img_cols = 150, 125

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255


## Set up `tfe.serving.QueueClient`

config = tfe.RemoteConfig.load("/tmp/tfe.config")

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())


input_shape = (1, 150, 125, 1)
output_shape = (1, 2)


client = tfe.serving.QueueClient(
    input_shape=input_shape,
    output_shape=output_shape)


sess = tfe.Session(config=config)


# ## Query Model

# User inputs
num_tests = 25
images, expected_labels = x_test[100:num_tests+100], y_test[100:num_tests+100]


for image, expected_label in zip(images, expected_labels):
    
    res = client.run(
        sess,
        image.reshape(1, 150, 125, 1))
    
    predicted_label = np.argmax(res)
    
    print("The image had label {} and was {} classified as {}".format(
        expected_label,
        "correctly" if expected_label == predicted_label else "incorrectly",
        predicted_label))







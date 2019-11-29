
# coding: utf-8

# # Private Predictions with TFE Keras

# # Step 3: Private Prediction using TFE Keras - Serving (Client)
# 
# After training your model with normal Keras and securing it with TFE Keras, you are ready to request some private predictions.

# In[1]:


import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tensorflow.keras.datasets import mnist


# ## Data
# 
# Here, we preprocess our MNIST data. This is identical to how we preprocessed during training.

# In[2]:


import pickle
file = open('./normal_test.pkl', 'rb')
# dump information to that file
test_normal = pickle.load(file)
# close the file
file.close()

file = open('./pneumonia_test.pkl', 'rb')
# dump information to that file
test_pneumonia = pickle.load(file)
# close the file
file.close()


# In[3]:


y_normal_t = np.zeros((234,), dtype=int)
y_pneumonia_t = np.ones((390,), dtype=int)

x_test = np.concatenate((test_normal, test_pneumonia))
y_test = np.concatenate((y_normal_t, y_pneumonia_t))
from sklearn.utils import shuffle
x_test, y_test = shuffle(x_test, y_test, random_state=0)


# In[4]:


# input image dimensions
img_rows, img_cols = 150, 125

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255


# In[5]:



# ## Set up `tfe.serving.QueueClient`
# 
# 
# Before querying the model, we just have to connect to it. To do so, we can create a client with `tfe.serving.QueueClient`. This creates a TFE queueing server on the client side that connects to the queueing server set up by `tfe.serving.QueueServer` in **Secure Model Serving**. The queue will be responsible for secretly sharing the plaintext data before submitting the shares in a prediction request.
# 
# Note that we have to use the same configuration as used by the server, including player configuration and protocol.

# In[6]:


config = tfe.RemoteConfig.load("/tmp/tfe.config")

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())


# In[7]:


input_shape = (1, 150, 125, 1)
output_shape = (1, 2)


# In[8]:


client = tfe.serving.QueueClient(
    input_shape=input_shape,
    output_shape=output_shape)


# In[9]:


sess = tfe.Session(config=config)


# ## Query Model
# 
# You are ready to get some private predictions! Calling `client.run` will insert the image into the queue created above, secret share the data locally, and submit the shares to the model server in **Secure Model Serving**.

# In[10]:


# User inputs
num_tests = 25
images, expected_labels = x_test[100:num_tests+100], y_test[100:num_tests+100]


# In[11]:


for image, expected_label in zip(images, expected_labels):
    
    res = client.run(
        sess,
        image.reshape(1, 150, 125, 1))
    
    predicted_label = np.argmax(res)
    
    print("The image had label {} and was {} classified as {}".format(
        expected_label,
        "correctly" if expected_label == predicted_label else "incorrectly",
        predicted_label))


# We are able to classify these three images correctly! But what's special about these predictions is that we haven't revealed any private information to get this service. The model host never saw your input data or your predictions, and you never downloaded the model. You were able to get private predictions on encrypted data with an encrypted model!

# In[ ]:





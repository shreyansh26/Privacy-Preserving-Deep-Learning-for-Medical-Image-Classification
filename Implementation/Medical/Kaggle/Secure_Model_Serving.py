
# coding: utf-8

# # Private Predictions with TFE Keras

# # Part 2: Secure Model Serving with TFE Keras

# Now that we have a trained model with normal Keras, we are ready to serve some private predictions. We can do that using TFE Keras.
# 
# To secure and serve this model, we will need three TFE servers. This is because TF Encrypted under the hood uses an encryption technique called multi-party computation (MPC). The idea is to split the model weights and input data into shares, then send a share of each value to the different servers. The key property is that if you look at the share on one server, it reveals nothing about the original value (input data or model weights).
# 
# To learn more about MPC, you can read this excellent [blog](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/).
# 
# In this notebook, you will be able serve private predictions after a series of simple steps:
# * Configure TFE Protocol to secure the model via secret sharing.
# * Launch three TFE servers.
# * Convert the TF Keras model into a TFE Keras model using `tfe.keras.models.clone_model`.
# * Serve the secured model using `tfe.serving.QueueServer`.

# In[1]:


from collections import OrderedDict

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
import tf_encrypted.keras.backend as KE


# ## Define Model
# 
# As you can see, we define almost the exact same model as before, except we provide a batch_input_shape. This allows TF Encrypted to better optimize the secure computations via predefined tensor shapes. For this MNIST demo, we'll send input data with the shape of (1, 28, 28, 1). We also return the logit instead of softmax because this operation is complex to perform using MPC, and we don't need it to serve prediction requests.

# In[2]:
import pickle
  
with open("./kernel_test_x.pkl", "rb") as f:
  y_train = pickle.load(f)
  
with open("./kernel_test_y.pkl", "rb") as f:
  y_valid = pickle.load(f)

num_classes = 2
input_shape = (1, 224, 224, 3)


# In[3]:

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam,RMSprop,SGD
model = Sequential()
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', batch_input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dense(2, name='logit'))

# With `load_weights` we can easily load the weights you have saved previously after training your model.

# In[4]:


pre_trained_weights = 'best_model_todate.h5'
model.load_weights(pre_trained_weights)


# ## Protocol
# 
# 
# 
# We first configure the protocol we will be using, as well as the servers on which we want to run it. We will be using the SecureNN protocol to secret share the model between each of the three TFE servers. Most importantly, this will add the capability of providing predictions on encrypted data.
# 
# Note that the configuration is saved to file as we will be needing it in the client as well.

# In[5]:

players = OrderedDict([
    ('server0', 'localhost:4000'),
    ('server1', 'localhost:4001'),
    ('server2', 'localhost:4002'),
])

config = tfe.RemoteConfig(players)
config.save('/tmp/tfe.config')


# In[6]:


tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())


# ## Launching servers
# 
# Before actually serving the computation below we need to launch TFE servers in new processes. Run the following in three different terminals. You may have to allow Python to accept incoming connections.

# In[7]:


for player_name in players.keys():
    print("python3.6 -m tf_encrypted.player --config /tmp/tfe.config {}".format(player_name))

input()
# ## Convert TF Keras into TFE Keras
# 
# Thanks to `tfe.keras.models.clone_model` you can convert automatically the TF Keras model into a TFE Keras model.

# In[ ]:


tf.reset_default_graph()
with tfe.protocol.SecureNN():
    tfe_model = tfe.keras.models.clone_model(model)


# ## Set up a new `tfe.serving.QueueServer`
# 
# `tfe.serving.QueueServer` will launch a serving queue, so that the TFE servers can accept prediction requests on the secured model from external clients.

# In[10]:
print("Model conversion done")
exit()
# Set up a new tfe.serving.QueueServer for the shared TFE model
q_input_shape = (1, 224, 224, 3)
q_output_shape = (1, 2)

server = tfe.serving.QueueServer(
    input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model
)


# ## Start Server
# 
# Perfect! with all of the above in place we can finally connect to our servers, push our TensorFlow graph to them, and start serving the model. You can set num_requests to set a limit on the number of predictions requests served by the model; if not specified then the model will be served until interrupted.

# In[11]:

sess = KE.get_session()


# In[ ]:

print("Waiting for queries for prediction")

request_ix = 1

def step_fn():
    global request_ix
    print("Served encrypted prediction {i} to client.".format(i=request_ix))
    request_ix += 1

# num_steps=3

server.run(
    sess,
    step_fn=step_fn)


# ## Cleanup!

# In[ ]:

'''
process_ids = get_ipython().getoutput(u"ps aux | grep '[p]ython3.6 -m tf_encrypted.player --config' | awk '{print $2}'")
for process_id in process_ids:
    get_ipython().system(u'kill {process_id}')
    print("Process ID {id} has been killed.".format(id=process_id))
'''

# In[ ]:





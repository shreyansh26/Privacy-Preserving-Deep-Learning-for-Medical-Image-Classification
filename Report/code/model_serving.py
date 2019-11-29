# coding: utf-8
"""
Private Predictions with TFE Keras
"""

from collections import OrderedDict

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
import tf_encrypted.keras.backend as KE


num_classes = 2
input_shape = (1, 150, 125, 1)

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 batch_input_shape=input_shape),
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
          tf.keras.layers.Dense(2, name='logit')
  ])

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 batch_input_shape=input_shape),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(num_classes, name='logit')
  ])


# With `load_weights` we can easily load the weights you have saved previously after training your model.

# ### Only for Differential Privacy model

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
    
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

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


pre_trained_weights = 'model_simple_dp.h5'
model.load_weights(pre_trained_weights)


# ## Protocol
#  
# We first configure the protocol we will be using, as well as the servers on which we want to run it. We will be using the SecureNN protocol to secret share the model between each of the three TFE servers. Most importantly, this will add the capability of providing predictions on encrypted data.
# 
# Note that the configuration is saved to file as we will be needing it in the client as well.

players = OrderedDict([
    ('server0', 'localhost:4000'),
    ('server1', 'localhost:4001'),
    ('server2', 'localhost:4002'),
])

config = tfe.RemoteConfig(players)
config.save('/tmp/tfe.config')


tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())


# ## Launching servers
# 
# Before actually serving the computation below we need to launch TFE servers in new processes. Run the following in three different terminals. You may have to allow Python to accept incoming connections.

for player_name in players.keys():
    print("python3.6 -m tf_encrypted.player --config /tmp/tfe.config {}".format(player_name))


# ## Convert TF Keras into TFE Keras
# 
# `tfe.keras.models.clone_model` can convert automatically the TF Keras model into a TFE Keras model.

tf.reset_default_graph()
with tfe.protocol.SecureNN():
    tfe_model = tfe.keras.models.clone_model(model)


# ## Set up a new `tfe.serving.QueueServer`
# 
# `tfe.serving.QueueServer` will launch a serving queue, so that the TFE servers can accept prediction requests on the secured model from external clients.

# Set up a new tfe.serving.QueueServer for the shared TFE model
q_input_shape = (1, 150, 125, 1)
q_output_shape = (1, 2)

server = tfe.serving.QueueServer(
    input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model
)


# ## Start Server
# 
sess = KE.get_session()


request_ix = 1

def step_fn():
    global request_ix
    print("Served encrypted prediction {i} to client.".format(i=request_ix))
    request_ix += 1

server.run(
    sess,
    step_fn=step_fn)



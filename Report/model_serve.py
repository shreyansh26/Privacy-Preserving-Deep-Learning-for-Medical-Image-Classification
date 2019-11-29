import tf_encrypted as tfe
import tf_encrypted.keras.backend as KE

# Load model weights trained from normal Keras
pre_trained_weights = 'model_simple.h5'
model.load_weights(pre_trained_weights)

# Configure the protocol
players = OrderedDict([
    ('server0', 'localhost:4000'),
    ('server1', 'localhost:4001'),
    ('server2', 'localhost:4002'),
])

config = tfe.RemoteConfig(players)
config.save('/tmp/tfe.config')

# Define the protocol to use - SecureNN
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

# Run in separate terminals to launch TFE servers
for player_name in players.keys():
    print("python3.6 -m tf_encrypted.player --config /tmp/tfe.config {}".format(player_name))


# Clone the Keras model to a TFE model
tf.reset_default_graph()
with tfe.protocol.SecureNN():
    tfe_model = tfe.keras.models.clone_model(model)


# Set up a new tfe.serving.QueueServer for the shared TFE model
q_input_shape = (1, 150, 125, 1)
q_output_shape = (1, 2)

server = tfe.serving.QueueServer(
    input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model
)


sess = KE.get_session()

# Wait for incoming requests for predictions and provide predictions when they arrive
request_ix = 1

def step_fn():
    global request_ix
    print("Served encrypted prediction {i} to client.".format(i=request_ix))
    request_ix += 1

# num_steps=3

server.run(
    sess,
    step_fn=step_fn)

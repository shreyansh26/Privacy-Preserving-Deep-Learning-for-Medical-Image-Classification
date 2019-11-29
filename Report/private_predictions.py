# Load the saved config file
config = tfe.RemoteConfig.load("/tmp/tfe.config")

# Set the config for TFE
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

# Set up tfe.serving.QueueClient
input_shape = (1, 150, 125, 1)
output_shape = (1, 2)

client = tfe.serving.QueueClient(
    input_shape=input_shape,
    output_shape=output_shape)

# Set the config for the session
sess = tfe.Session(config=config)


# Query the model
# User inputs
num_tests = 25
images, expected_labels = x_test[100:num_tests+100], y_test[100:num_tests+100]

for image, expected_label in zip(images, expected_labels):
    # Get predictions form the model
    res = client.run(
        sess,
        image.reshape(1, 150, 125, 1))
    
    predicted_label = np.argmax(res)
    # Display the results
    print("The image had label {} and was {} classified as {}".format(
        expected_label,
        "correctly" if expected_label == predicted_label else "incorrectly",
        predicted_label))

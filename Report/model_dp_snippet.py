from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

dpsgd = True            # If True, train with DP-SGD
learning_rate = 0.15    # Learning rate for training
noise_multiplier = 1.1  # Ratio of the standard deviation to the clipping norm
l2_norm_clip = 1.0      # Clipping norm
batch_size = 250        # Batch size
epochs = 20             # Number of epochs
microbatches = 50       # Number of microbatches


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
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


tf.logging.set_verbosity(tf.logging.INFO)
if dpsgd and batch_size % microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=(28, 28, 1)),
      tf.keras.layers.AveragePooling2D(2, 1),
      tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
      tf.keras.layers.AveragePooling2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

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

# Train model with Keras
model.fit(train_data, train_labels,
        epochs=epochs,
        validation_data=(test_data, test_labels),
        batch_size=batch_size)

# Compute the privacy budget expended.
if dpsgd:
    eps = compute_epsilon(epochs * 60000 // batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
else:
    print('Trained with vanilla non-private SGD optimizer')

# Define model
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
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(2, activation='softmax')
  ])


# Define constants
batch_size = 32
epochs = 40


# Save model checkpoint for best model yet in terms of highest validation accuracy
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(filepath='./models/model_simple.h5',monitor="val_acc", save_best_only=True, save_weights_only=False)

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           callbacks=[mcp],
           validation_data=(x_test, y_test))

# Evaluate model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

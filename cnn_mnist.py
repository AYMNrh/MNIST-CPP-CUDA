### Setting up

# All the imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time

# Record start time
start_time = time.time()

# Supress deprecation warnings
import logging
logging.getLogger('tensorflow').disabled = True

# Fetch "Fashion MNIST" data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# A good rule of thumb is to normalise input values - i.e. transform them to a
# scale of 0 to 1. Each element in this dataset is a pixel value of 0 to 255, so
# we'll normalise / rescale these values.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Map for human readable class names
class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']


print("Shape of Training Image Data: " + str(x_train.shape))
print("Shape of Training Class Data: " + str(y_train.shape))
print("Shape of Test Image Data: " + str(x_test.shape))
print("Shape of Test Class Data: " + str(y_test.shape))

### Create and build Convolutional Neural Network

# We begin by defining the a empty stack. We'll use this for building our
# network, later by layer.
model = tf.keras.models.Sequential()

# We start with a convolutional layer this will extract features from
# the input images by sliding a convolution filter over the input image,
# resulting in a feature map.
model.add(
    tf.keras.layers.Conv2D(
        filters=32, # How many filters we will learn
        kernel_size=(3, 3), # Size of feature map that will slide over image
        strides=(1, 1), # How the feature map "steps" across the image
        padding='valid', # We are not using padding
        activation='relu', # Rectified Linear Unit Activation Function
        input_shape=(28, 28, 1) # The expected input shape for this layer
    )
)

# The next layer we will add is a Maxpooling layer. This will reduce the
# dimensionality of each feature, which reduces the number of parameters that
# the model needs to learn, which shortens training time.
model.add(
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), # Size feature will be mapped to
        strides=(2, 2) # How the pool "steps" across the feature
    )
)

# Output from previous layer is a 3D tensor. This must be flattened to a 1D
# vector before being fed to the Dense Layers.
model.add(
    tf.keras.layers.Flatten()
)

# Final layer with 10 outputs and a softmax activation. Softmax activation
# enables me to calculate the output based on the probabilities.
# Each class is assigned a probability and the class with the maximum
# probability is the model’s output for the input.
model.add(
    tf.keras.layers.Dense(
        units=10, # Output shape
        activation='softmax' # Softmax Activation Function
    )
)

# Build the model
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, # loss function
    optimizer=tf.keras.optimizers.Adam(), # optimizer function
    metrics=['accuracy'] # reporting metric
)

# Display a summary of the models structure
model.summary()

### Visualise the Model

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

### Train the CNN

# Add an empty color dimension as the Convolutional net is expecting this
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Train the CNN on the training data
history = model.fit(

      # Training data : features (images) and classes.
      x_train, y_train,

      # number of samples to work through before updating the
      # internal model parameters via back propagation.
      batch_size=256,

      # An epoch is an iteration over the entire training data.
      epochs=10,

      # The model will set apart his fraction of the training
      # data, will not train on it, and will evaluate the loss
      # and any model metrics on this data at the end of
      # each epoch.
      validation_split=0.2,

      verbose=1)

### Evaluate model with test data and view results

# Get Model Predictions for test data
predicted_classes = model.predict(x_test)
predicted_classes = np.argmax(predicted_classes,axis=1)
print(classification_report(y_test, predicted_classes, target_names=class_names))

### View examples of incorrectly classified test data

incorrect = np.nonzero(predicted_classes!=y_test)[0]

# Display the first 16 incorrectly classified images from the test data set
plt.figure(figsize=(15, 8))
for j, incorrect in enumerate(incorrect[0:8]):
    plt.subplot(2, 4, j+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap="Reds")
    plt.title("Predicted: {}".format(class_names[predicted_classes[incorrect]]))
    plt.xlabel("Actual: {}".format(class_names[y_test[incorrect]]))


# The layer we want to copy from the trained CNN
layer_name = 'conv2d'

# Get the list of layers for the existing model
layer_dict = {layer.name : layer for layer in model.layers}

# Create a copy of our existing model containing just the Conv2D Layer
modelslice = tf.keras.Model(inputs=model.inputs, outputs=layer_dict[layer_name].output)

# Choose an image (0 to 59999) from the training set
image = x_train[0]

# Add the extra dimension expected by the slice
image = np.expand_dims(image, axis=0)

# Send the image through the model
feature_maps = modelslice.predict(image)

plt.figure(figsize=(15, 8))

# We are assuming that we have 32 feature maps
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(feature_maps[0, :, :, i-1], cmap=plt.cm.binary)

# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Elapsed time : {:.2f} seconds".format(elapsed_time))
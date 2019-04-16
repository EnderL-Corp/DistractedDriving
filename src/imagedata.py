from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow import keras as k

import numpy as np
import matplotlib.pyplot as plt

class_names = ['Normal driving',
               'Texting',
               'Talking on phone',
               'Operating Radio',
               'Drinking',
               'Reaching behind',
               'Hair and makeup',
               'Talking to passenger']


def train_neural_net():
    # Set checkpoint data so we can save the state of our neural net to load later
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = k.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

    distracted_drivers = 0
    # distracted_drivers = csvreader.parse()
    (train_images, train_labels), (test_images, test_labels) = distracted_drivers  # .getData()

    # Covert greyscale images with pixel values from 0-255 to pixel values 0-1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define our model. This model consists of 3 layers:
    # Flatten converts the image into a one dimensional array to pass through our net
    # The second layer has 128 nodes that are all connected (Dense)
    # The third layer has 10 nodes whose values will be probabilities that sum to one. These represent
    #       The confidence the model has that a certain image fits a certain label
    model = k.Sequential([
        k.layers.Flatten(input_shape=(28, 28)),
        k.layers.Dense(128, activation=tf.nn.relu),
        k.layers.Dense(8, activation=tf.nn.softmax)
    ])

    # Set the optimizer, loss, and metric our model will use while tuning itself
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train our model, using the training images and labels
    model.fit(train_images,
              train_labels,
              epochs=5,
              callbacks=[checkpoint_callback])

    # Save our model again, but this time the entire model in HDF5 format
    model.save('distracted_driver_recognition.h5')

    # Apply our neural net to the test data and see how it performs
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)


from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow import keras as k

import numpy as np
import matplotlib.pyplot as plot

from src import imgreader

class_lookup_table = [['c0', 'Normal driving'],
                      ['c1', 'Texting'],
                      ['c2', 'Talking on phone'],
                      ['c3', 'Texting'],
                      ['c4', 'Talking on phone'],
                      ['c5', 'Operating Radio'],
                      ['c6', 'Drinking'],
                      ['c7', 'Reaching behind'],
                      ['c8', 'Hair and makeup'],
                      ['c9', 'Talking to passenger']]

class_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']


def train_neural_net():
    # Set checkpoint data so we can save the state of our neural net to load later
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = k.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

    distracted_drivers = imgreader.getTrainFiles()
    print("Got files")
    train_images, train_labels = distracted_drivers  # .getData()

    # Covert greyscale images with pixel values from 0-255 to pixel values 0-1
    train_images = train_images / 255.0

    # Define our model. This model consists of 3 layers:
    # Flatten converts the image into a one dimensional array to pass through our net
    # The second layer has 128 nodes that are all connected (Dense)
    # The third layer has 10 nodes whose values will be probabilities that sum to one. These represent
    #       The confidence the model has that a certain image fits a certain label
    model = k.Sequential([
        k.layers.Flatten(input_shape=(640, 480)),
        k.layers.Dense(128, activation=tf.nn.relu),
        k.layers.Dense(8, activation=tf.nn.softmax)
    ])
    print("Created net")

    # Set the optimizer, loss, and metric our model will use while tuning itself
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Abt to train")
    # Train our model, using the training images and labels
    model.fit(train_images,
              train_labels,
              epochs=5,
              callbacks=[checkpoint_callback])
    print("Trained")

    # Save our model again, but this time the entire model in HDF5 format
    model.save('distracted_driver_recognition.h5')

    """# Apply our neural net to the test data and see how it performs
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # Save our model's guesses
    predictions = model.predict(test_images)

    # Show a cool plot of some results
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plot.figure(figsize=(4 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plot.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plot.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plot.show()
    """


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])

    plot.imshow(img, cmap=plot.cm)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100 * np.max(predictions_array),
                                          class_names[true_label]),
                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])
    this_plot = plot.bar(range(10), predictions_array, color="#777777")
    plot.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')

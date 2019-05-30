from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras as k

import numpy as np
import matplotlib.pyplot as plot

import src.imgreader as imgreader

available_subjects = [2, 14, 22, 24, 26, 42,
                      45, 47, 52, 61, 66, 81]

                      # [2, 12, 14, 15, 16, 21, 22, 24, 26, 35, 41, 42,
                      # 45, 47, 49, 50, 51, 52, 56, 61, 64, 66, 75, 81]
                      # p039 and p072 aint it

class_lookup_table = {'c0': 'Normal driving',
                      'c1': 'Distracted Driving',       # 'Testing'
                      'c2': 'Talking on phone',
                      'c3': 'Texting',
                      'c4': 'Talking on phone',
                      'c5': 'Operating radio',
                      'c6': 'Drinking or eating',
                      'c7': 'Reaching behind',
                      'c8': 'Hair and makeup',
                      'c9': 'Talking to passenger'}

class_names = ['c0', 'c1']


def train_neural_net():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    # Define our model. This model consists of 4 layers:
    # 2 2dConvolution layers apply spacial convolution over each image
    # The third layer flattens the image into a one dimensional array to pass through our net
    # The fourth layer has 10 nodes whose values will be probabilities that sum to one. These represent
    #       The confidence the model has that a certain image fits a certain label
    '''
    model = k.Sequential([
        k.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(240, 320, 3), activation='relu'),
        # k.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        k.layers.MaxPooling2D(pool_size=(2, 2)),

        k.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        # k.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        k.layers.MaxPooling2D(pool_size=(2, 2)),

        k.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        # k.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        k.layers.MaxPooling2D(pool_size=(2, 2)),

        k.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        #k.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        k.layers.MaxPooling2D(pool_size=(2, 2)),

        k.layers.Flatten(),
        k.layers.Dense(256, activation='relu'),
        k.layers.Dropout(0.5),

        k.layers.Dense(256, activation='relu'),
        k.layers.Dropout(0.5),

        k.layers.Dense(2),
        k.layers.Activation(activation='softmax')
    ])
    '''

    model = k.Sequential([
        k.layers.Conv2D(32, (3, 3), input_shape=(240, 320, 3)),
        k.layers.Activation("relu"),
        k.layers.MaxPooling2D(pool_size=(2, 2)),

        k.layers.Conv2D(32, (3, 3)),
        k.layers.Activation("relu"),
        k.layers.MaxPooling2D(pool_size=(2, 2)),

        k.layers.Flatten(),
        k.layers.Dense(16),
        k.layers.Activation("relu"),
        k.layers.Dropout(0.5),
        k.layers.Dense(2),
        k.layers.Activation("softmax")
    ])

    print("[imagedata.train_neural_net]: Created neural net")
    model.summary()

    # Set the optimizer, loss, and metric our model will use while tuning itself
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[imagedata.train_neural_net]: Compiled neural net")

    for round in range(0, 2):
        print(f"Training for round {round}")

        for subject in available_subjects:
            # Retrieve our images for the subject we will be looking at
            distracted_drivers = imgreader.get_subject_data(subject)

            print(f"[imagedata.train_neural_net]: Retrieved files for subject {subject}")
            (train_images, train_labels) = distracted_drivers

            print(np.array(train_images).shape)
            # Reshape the list of our training images to correctly be used by the neural net
            train_images = train_images.reshape(len(train_images), 240, 320, 3)
            print(np.array(train_images).shape)

            print(f"[imagedata.train_neural_net]: Training for subject {subject}")
            # Train our model, using the training images and labels
            model.fit(train_images,
                      train_labels,
                      epochs=3)

            print(f"[imagedata.train_neural_net]: Trained for subject {subject}")

    # Save our model to recover later
    model.save('distracted_driver_recognition.h5')


def test_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    trained_model = k.models.load_model('distracted_driver_recognition.h5')

    distracted_drivers = imgreader.get_train_data_for_testing()
    print("[imagedata.train_neural_net]: Got files for subjects")

    test_images = distracted_drivers

    test_images_new = test_images.reshape(len(test_images), 240, 320, 3)

    # Save our model's guesses
    predictions = trained_model.predict(test_images_new)

    # Show a cool plot of some results
    num_rows = 3
    num_cols = 8
    num_images = num_rows * num_cols
    plot.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plot.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_images)
        plot.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions)
    plot.show()


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])

    plot.imshow(img, cmap=plot.cm.binary)

    predicted_label = np.argmax(predictions_array)

    plot.xlabel("{} {:2.0f}%".format(class_lookup_table.get(class_names[predicted_label]),
                                     100 * np.max(predictions_array)))


def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    simple_prediction = [predictions_array[0], 0]
    for i in range(len(predictions_array) - 1):
        simple_prediction[1] += predictions_array[i + 1]
    plot.grid(False)
    plot.xticks([])
    plot.yticks([])
    this_plot = plot.bar(range(2), simple_prediction, color="#777777")
    plot.ylim([0, 1])
    predicted_label = np.argmax(simple_prediction)

    this_plot[predicted_label].set_color('red')


if __name__ == "__main__":
    test_model()

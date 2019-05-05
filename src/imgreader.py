import numpy as np
import os
import random
import cv2
import csv
from pathlib import Path
import sys


def get_train_files():
    """
    Method generates a csv containing the train files and labels
    Don't use this unless you want to waste 3 hours of your life only to generate a 30 gig file
    """
    if Path('D:\\Python Projects\\DistractedDriving\\imagelist.csv').is_file():
        return
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    i = 0
    imagelist = np.genfromtxt(dir + 'driver_imgs_list.csv', delimiter=',', dtype='U')
    np.set_printoptions(threshold=np.nan)
    with open('D:\\Python Projects\\DistractedDriving\\imagelist.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        for distractionfolder in os.listdir(imgdir):
            for image in os.listdir(imgdir + '\\' + distractionfolder):
                img = cv2.imread(imgdir + '\\' + distractionfolder + '\\' + image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #x_train = np.append(x_train, img)
                (labelx, labely) = np.where(imagelist == image)
                labelx = labelx[0]
                labely = 1
                #y_train = np.append(y_train, imagelist[labelx][labely])
                filewriter.writerow([img, imagelist[labelx][labely]])
                i += 1
                print(i)
# D:\\Pictures\\Distracted Driving\\imgs\\train


def get_subject_data(subj=2):
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    x_train = [[[]]]
    y_train = np.array([])
    i = 0
    j = 0
    np.set_printoptions(threshold=sys.maxsize)
    imagelist = np.genfromtxt(dir + 'driver_imgs_list.csv', delimiter=',', dtype='U')
    subj = str(subj)
    while len(subj) < 3:
        subj = '0' + subj
    subj = 'p' + subj
    datapts = np.where(imagelist == subj)[0]
    print(f'[imgreader.get_subject_data]: Retrieving data on subject {subj}... This may take a while')
    first = datapts[0]
    last = datapts[np.size(datapts) - 1]
    for locus in datapts:
        j += 1
        if j % 5 == 0:
            continue
        label = imagelist[locus][1]
        image = cv2.imread(imgdir + '\\' + label + '\\' + imagelist[locus][2])  # Load image from file
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to greyscale
        image = cv2.resize(image, (int(image.shape[1]*0.5), int(image.shape[0]*0.5)))
        image = (np.divide(np.array(image), 255)).tolist()  # Convert pixel values from 0-255 to 0-1
        x_train.insert(i, image)
        y_train = np.append(y_train, str(label).lstrip('c'))
        i += 1
        draw_progress_bar((locus-first)/(last-first))

    del x_train[i]
    print('\n')

    return np.array(x_train), y_train


def get_test_subject_data():
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\test'
    x_test = [[[]]]
    np.set_printoptions(threshold=sys.maxsize)
    i = 0
    for x in range(0, 24):
        image = cv2.imread(imgdir + '//' + random.choice(os.listdir(imgdir)))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
        image = (np.divide(np.array(image), 255)).tolist()  # Convert pixel values from 0-255 to 0-1
        x_test.insert(i, image)
        i += 1
    del x_test[i]

    return np.array(x_test)


def get_train_data_for_testing():
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    x_test = [[[]]]
    np.set_printoptions(threshold=sys.maxsize)
    i = 0
    for x in range(0, 24):
        newdir = imgdir + '\\c' + str(x % 10)
        image = cv2.imread(newdir + '\\' + random.choice(os.listdir(newdir)))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
        image = (np.divide(np.array(image), 255)).tolist()  # Convert pixel values from 0-255 to 0-1
        x_test.insert(i, image)
        i += 1
    del x_test[i]

    return np.array(x_test)


# progress bar method i found on stack overflow :)
def draw_progress_bar(percent, length=50):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(length * percent), length, percent * 100))
    sys.stdout.flush()


def testlist():
    dir='D:\\Pictures\\Distracted Driving\\imgs\\test\\img_1.jpg'
    img=cv2.imread(dir)
    print(img.shape)


def edge_det_test():
    image = cv2.imread('D:\\Pictures\\Distracted Driving\\imgs\\test\\img_385.jpg')
    edge = cv2.Canny(image, 100, 100)
    cv2.imshow('image', image)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Testing
    testlist()
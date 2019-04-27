import numpy as np
import os
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

'''
def get_subject_data(subj=2):
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    x_train = np.empty(dtype='U', shape=0)
    y_train = np.empty(dtype='U', shape=0)
    np.set_printoptions(threshold=sys.maxsize)
    imagelist = np.genfromtxt(dir + 'driver_imgs_list.csv', delimiter=',', dtype='U')
    subj = str(subj)
    while len(subj) < 3:
        subj = '0' + subj
    subj = 'p' + subj
    print('imgreader.get_subject_data: Gathering data on subject ' + subj)
    datapts = np.where(imagelist == subj)[0]
    print('imgreader.get_subject_data: Working... this may take up to 5 minutes')
    first = datapts[0]
    last = datapts[np.size(datapts) - 1]
    for locus in datapts:
        label = imagelist[locus][1]
        image = cv2.imread(imgdir + '\\' + label + '\\' + imagelist[locus][2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (int(image.shape[1]*0.25), int(image.shape[0]*0.25)))
        x_train = np.append(x_train, image)
        y_train = np.append(y_train, label)
        draw_progress_bar((locus-first)/(last-first))

    print(x_train[0])
    return x_train, y_train
'''


def get_subject_data(subj=2):
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    x_train = [[[]]]
    y_train = [[[]]]
    i=0
    np.set_printoptions(threshold=sys.maxsize)
    imagelist = np.genfromtxt(dir + 'driver_imgs_list.csv', delimiter=',', dtype='U')
    subj = str(subj)
    while len(subj) < 3:
        subj = '0' + subj
    subj = 'p' + subj
    print('imgreader.get_subject_data: Gathering data on subject ' + subj)
    datapts = np.where(imagelist == subj)[0]
    print('imgreader.get_subject_data: Working... this may take up to 5 minutes')
    first = datapts[0]
    last = datapts[np.size(datapts) - 1]
    for locus in datapts:
        label = imagelist[locus][1]
        image = cv2.imread(imgdir + '\\' + label + '\\' + imagelist[locus][2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (int(image.shape[1]*0.25), int(image.shape[0]*0.25)))
        x_train.insert(i, image.tolist())
        y_train.insert(i, label)
        i += 1
        draw_progress_bar((locus-first)/(last-first))

    del x_train[i]

    return x_train, y_train


# progress bar method i found on stack overflow :)
def draw_progress_bar(percent, length=50):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(length * percent), length, percent * 100))
    sys.stdout.flush()


def testlist():
    if Path('D:\\Python Projects\\DistractedDriving\\imagelist.csv').is_file() == True:
        with open('D:\\Python Projects\\DistractedDriving\\imagelist.csv', 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                print(row[1])


if __name__ == "__main__":
    # Testing
    get_subject_data(15)

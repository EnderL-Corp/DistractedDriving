import numpy as np
import os
import cv2
import csv
from pathlib import Path


def get_train_files():
    """
    Method returns a 2D numpy array containing the train files and labels
    """

    if Path('D:\\Python Projects\\DistractedDriving\\imagelist.csv').is_file():
        return

    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    # x_train = np.empty(dtype=None, shape=0)
    # y_train = np.empty(dtype=None, shape=0)
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
    # return x_train, y_train


def get_subject_data(subj=2):
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    i = 0
    x_train = np.empty(dtype=None, shape=0)
    y_train = np.empty(dtype=None, shape=0)
    imagelist = np.genfromtxt(dir + 'driver_imgs_list.csv', delimiter=',', dtype='U')
    for distractionfolder in os.listdir(imgdir):
        for image in os.listdir(imgdir + '\\' + distractionfolder):
            img = cv2.imread(imgdir + '\\' + distractionfolder + '\\' + image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_train = np.append(x_train, img)
            (labelx, labely) = np.where(imagelist == image)
            labelx = labelx[0]
            labely = 1
            y_train = np.append(y_train, imagelist[labelx][labely])
            i += 1
            print(i)


i=np.where[0][0]
j=0
while arr[i+j][0] == subj:
    type=arr[i+j][1]
    


def testlist():
    if Path('D:\\Python Projects\\DistractedDriving\\imagelist.csv').is_file() == True:
        with open('D:\\Python Projects\\DistractedDriving\\imagelist.csv', 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                print(row[1])


if __name__ == "__main__":
    get_train_files()

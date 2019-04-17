import numpy as np
import os
import cv2


def getTrainFiles():
    """
    Method returns a 2D numpy array containing the train files and labels
    """
    dir = 'D:\\Pictures\\Distracted Driving\\'
    imgdir = dir + 'imgs\\train'
    x_train = np.empty(dtype=None, shape=0)
    y_train = np.empty(dtype=None, shape=0)
    imagelist = np.genfromtxt(dir + 'driver_imgs_list.csv', delimiter=',', dtype='U')
    for distractionfolder in os.listdir(imgdir):
        for image in os.listdir(imgdir + '\\' + distractionfolder):
            img = cv2.imread(imgdir + '\\' + distractionfolder + '\\' + image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_train = np.append(x_train, img)
            (labelx, labely) = np.where(imagelist == image)
            labelx = labelx[0][0]
            labely = 1
            y_train = np.append(y_train, imagelist[labelx][labely])
    return x_train, y_train

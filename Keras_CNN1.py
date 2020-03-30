import keras
import numpy as np
from tqdm import tqdm
import os
import cv2

from random import shuffle
from tools import getPlateInfo, dataAugmentFormat, augment, prepareDataset, showPrediction
from datetime import datetime
from keras.metrics import MeanIoU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keras.backend as K

from albumentations import Resize

def iou(y_true, y_pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    iou = 0
    for i in range(K.int_shape(y_pred)[0]):
        boxA = y_pred[i]
        boxB = y_true[i]
        xA = K.max(boxA[0], boxB[0])
        yA = K.max(boxA[2], boxB[2])
        xB = K.min(boxA[1], boxB[1])
        yB = K.min(boxA[3], boxB[3])

        interArea = K.max(0, xB - xA + 1) * K.max(0, yB - yA + 1)

        boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
        boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

        iou += interArea / float(boxAArea + boxBArea - interArea)
    #MEAN
    mean = iou/K.int_shape(y_pred)[0]
    return 1-mean


penalty = 100

def lf(y_true,y_pred):
    mean_square = keras.losses.mean_squared_error(y_true[:,0:4], y_pred[:,0:4])
    check_class = np.subtract(y_true[:,4], y_pred[:,4])
    check_class = check_class * -penalty
    check_class = keras.backend.mean(check_class)
    return mean_square + check_class



# PREPARE THE DATASET FOR TRAINING
imageSize = [1920, 1080, 3]
trainingImageSize = [640, 360, 3]
path = "C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/training"
imageDirectory = os.listdir(path+"/images")
numberOfImages = len(imageDirectory)
category_id_to_name = {1: "License Plate"}
pairedData = [] # paired x and y, so that we may shuffle

if numberOfImages != len(os.listdir(path+"/images")):
    print("ERROR: The number of images is not equal to the number of labels")
    exit()

x_train, y_train = prepareDataset(path, 2500, trainingImageSize, 2700)


print("X =", x_train[0])
print(x_train[0].shape)
print("Y =", y_train[0])
print(y_train[0].shape)
"""
print("Y shape:", y_train[0].shape)
print("=>", y_train[0])

print("X shape:", x_train[0].shape)
"""

# PREPARE THE CNN1 FOR TRAINING
# CNN 1 - Detecting the License Plate
cnn1 = Sequential(name = "CNN1")
cnn1.add(Conv2D(64, (3, 3), input_shape=(trainingImageSize[1], trainingImageSize[0], trainingImageSize[2]), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(128, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(256, (3, 3), activation="relu"))
cnn1.add(Flatten())
cnn1.add(Dense(64, activation="relu"))
cnn1.add(Dropout(0.2))
cnn1.add(Dense(32, activation="relu"))
cnn1.add(Dropout(0.2))
cnn1.add(Dense(4))

cnn1.compile(optimizer='sgd', loss="mse", metrics=['accuracy'])
print("Compiled!")
#cnn1.summary()

print("Fitting!")
cnn1.fit(x = x_train, y = y_train, epochs = 10, batch_size = 10)

print("Saving!")
name = (datetime.now().strftime("model_%d-%m-%Y_%I-%M-%S_%p"))
cnn1.save("{}.h5".format(name))

toShow = 10
x_test, y_test = prepareDataset("C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/testing", 10, trainingImageSize, 500)

results = cnn1.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

pred = cnn1.predict(x = x_test[:toShow])*255

for i in range(toShow):
    predFormatted = dataAugmentFormat(x_test[i], pred[i])
    showPrediction(predFormatted)
    print("Prediction {}".format(i))
    print("Model Prediction =", pred[i])
    print("Correct Prediction =", y_test[i]*255)

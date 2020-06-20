import keras
import numpy as np
from tqdm import tqdm
import os
import cv2

from random import shuffle
from tools import getPlateInfo, dataAugmentFormat, augment, prepareDataset, showPrediction
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

trainingImageSize = [30, 15, 3]
path = "C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/trainingCNN2"
x_train, y_train = prepareDataset(path, 2000, trainingImageSize, 2000)

cnn2 = Sequential(name ="CNN2")
cnn2.add(Conv2D(64, (3, 3), input_shape=(trainingImageSize[1], trainingImageSize[0], trainingImageSize[2]), activation="relu"))
cnn2.add(MaxPooling2D(pool_size=(2, 2)))
cnn2.add(Conv2D(128, (3, 3), activation="relu"))
cnn2.add(MaxPooling2D(pool_size=(2, 2)))
cnn2.add(Conv2D(256, (3, 3), activation="relu"))
cnn2.add(Flatten())
cnn2.add(Dense(64, activation="relu"))
cnn2.add(Dropout(0.2))
cnn2.add(Dense(32, activation="relu"))
cnn2.add(Dropout(0.2))
cnn2.add(Dense(7*4))

cnn2.compile(optimizer='sgd', loss="mse", metrics=['accuracy'])
print("Compiled!")
#cnn1.summary()

print("Fitting!")
cnn2.fit(x = x_train, y = y_train, epochs = 10, batch_size = 10)

print("Saving!")
name = (datetime.now().strftime("model_%d-%m-%Y_%I-%M-%S_%p"))
cnn2.save("{}.h5".format(name))

toShow = 10
x_test, y_test = prepareDataset("C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/testing", 10, trainingImageSize, 500)

results = cnn2.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

pred = cnn2.predict(x = x_test[:toShow])*255

for i in range(toShow):
    predFormatted = dataAugmentFormat(x_test[i], pred[i])
    showPrediction(predFormatted)
    print("Prediction {}".format(i))
    print("Model Prediction =", pred[i])
    print("Correct Prediction =", y_test[i]*255)
import keras
import numpy as np
from tqdm import tqdm
import os
import cv2

from random import shuffle
from tools import getPlateInfo, dataAugmentFormat, augment, visualize, prepareDataset
from keras.models import load_model

from keras.metrics import MeanIoU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from albumentations import Resize


def showPrediction(coposition):
    visualize(coposition, category_id_to_name)
    #print('here')


# PREPARE THE DATASET FOR TRAINING
imageSize = [1920, 1080, 3]
trainingImageSize = [640, 360, 3]
path = "C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/testing"

category_id_to_name = {1: "License Plate"}

x_test, y_test = prepareDataset(path, 50, trainingImageSize, 500)

i = 1
predFormatted = dataAugmentFormat(x_test[i], y_test[i]*255)
resizedComposedImage = augment([Resize(p=1, height=144, width=256)])
finalComposedImage = resizedComposedImage(**predFormatted)
showPrediction(finalComposedImage)

model = load_model("model_30-03-2020_01-02-36_PM.h5")

results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)


# score = model.evaluate(x_train, y_train)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


toShow = 3

pred = model.predict(x = x_test[:toShow])*255

for i in range(toShow):
    predFormatted = dataAugmentFormat(x_test[i], pred[i])
    showPrediction(predFormatted)
    print("Prediction {}".format(i))
    print("Model Prediction =", pred[i])
    print("Correct Prediction =", y_test[i]*255)
    

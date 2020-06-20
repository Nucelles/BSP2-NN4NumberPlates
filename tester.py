import keras
import numpy as np
from tqdm import tqdm
import os
import cv2

from random import shuffle
from tools import getPlateInfo, dataAugmentFormat, dice_loss, prepareDataset, showPrediction, showModelGraphs
from datetime import datetime
from keras.metrics import MeanIoU
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keras.backend as K
import tensorflow as tf


def toFormat(bbox, imgSize):
    x_min, y_min, x_max, y_max = bbox
    normalizedLabel = [x_min * (imgSize[1] + 1),
                       y_min * (imgSize[0] + 1),
                       x_max * (imgSize[1] + 1),
                       y_max * (imgSize[0] + 1)]

    return normalizedLabel

# Activity E & F
def evaluateModel(model, imgSize, path):
    # Prepare the test dataset
    # Evaluate the dataset
    # Predicts 5 images randomly from the dataset

    toShow = 5
    x_test, y_test = prepareDataset(path, 400, imgSize, 400)

    results = model.evaluate(x_test, y_test, batch_size=10)
    print('test loss, test acc:', results)

    pred = model.predict(x=x_test[:toShow])

    for i in range(toShow):
        y_testPrediction = toFormat(y_test[i], imgSize=imgSize)
        modelPrediction = toFormat(pred[i], imgSize=imgSize)

        predFormatted = dataAugmentFormat(x_test[i], modelPrediction)
        showPrediction(predFormatted, y_testPrediction)

        print("Prediction {}".format(i+1))
        print("Model Prediction =", modelPrediction)
        print("                 =", pred[i])
        print("Correct Prediction =", y_testPrediction)
        print("                   =", y_test[i])


trainingImageSize = [512, 288, 3]
path = "D:/BiCS/BSP S2/Project/DATA/testing"
cnn1 = load_model("Models/model_15.h5")
evaluateModel(cnn1, trainingImageSize, path)










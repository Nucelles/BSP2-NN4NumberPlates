import keras
import numpy as np
from tqdm import tqdm
import os
import cv2

from random import shuffle
from tools import getPlateInfo, dataAugmentFormat, augment

from keras.metrics import MeanIoU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from albumentations import Resize


# PREPARE THE DATASET FOR TRAINING
imageSize = [1920, 1080, 3]
path = "C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/training"
imageDirectory = os.listdir(path+"/images")
numberOfImages = len(imageDirectory)
category_id_to_name = {1: "License Plate"}
pairedData = [] # paired x and y, so that we may shuffle

if numberOfImages != len(os.listdir(path+"/images")):
    print("ERROR: The number of images is not equal to the number of labels")
    exit()

for currImage in tqdm(range(1000)):

    # Prepare the X input
    imageName = imageDirectory[currImage]
    image = cv2.imread(path+"/images/"+imageName) / 255

    # Prepare the Y input
    label = imageName[:-3]+"txt"
    label = getPlateInfo(open(path+"/plates/"+label))[1]

    # Resize the image keeping the bbox location
    composedImage = dataAugmentFormat(image, label)
    resizedComposedImage = augment([Resize(p=1, height=144, width=256)])
    finalComposedImage = resizedComposedImage(**composedImage)

    dataPair = [finalComposedImage["image"], np.array(finalComposedImage["bboxes"][0])]
    pairedData.append(dataPair)

shuffle(pairedData)
x_train = np.array([a[0] for a in pairedData])
y_train = np.array([a[1] for a in pairedData])


print(y_train[0])


# PREPARE THE CNN1 FOR TRAINING
# CNN 1 - Detecting the License Plate
cnn1 = Sequential(name = "CNN1")
cnn1.add(Conv2D(32, (3, 3), input_shape=(144, 256, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(64, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(128, (3, 3), activation="relu"))
cnn1.add(Flatten())
cnn1.add(Dense(256, activation="relu"))
cnn1.add(Dropout(0.2))
cnn1.add(Dense(128, activation="relu"))
cnn1.add(Dense(4))

cnn1.compile(optimizer='sgd', loss='mse', metrics=['accuracy', MeanIoU(num_classes=1)])
print("Compiled!")
#cnn1.summary()

print("Fitting!")
cnn1.fit(x = x_train, y = y_train, epochs = 10, batch_size = 32)


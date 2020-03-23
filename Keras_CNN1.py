import keras
import numpy as np
from tqdm import tqdm
import os
import cv2
import pickle

from random import shuffle
from tools import getPlateInfo
from keras.metrics import MeanIoU

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D



# PREPARE THE DATASET FOR TRAINING
imageSize = [1920, 1080, 3]
path = "C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/training"
imageDirectory = os.listdir(path+"/images")
numberOfImages = len(imageDirectory)

pairedData = [] # paired x and y, so that we may shuffle

if numberOfImages != len(os.listdir(path+"/images")):
    print("ERROR: The number of images is not equal to the number of labels")
    exit()

for currImage in tqdm(range(200)):

    #Prepare the X input
    imageName = imageDirectory[currImage]
    image = cv2.imread(path+"/images/"+imageName) / 255


    #Prepare the Y input
    label = imageName[:-3]+"txt"
    label = getPlateInfo(open(path+"/plates/"+label))[1]

    pairedData.append([image, label])


shuffle(pairedData)

x_train = [a[0] for a in pairedData]
y_train = [a[1] for a in pairedData]




# PREPARE THE CNN1 FOR TRAINING
# CNN 1 - Detecting the License Plate
cnn1 = Sequential(name = "CNN1")
cnn1.add(Conv2D(32, (3, 3), input_shape=(1080, 1920, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(64, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(128, (3, 3), activation="relu"))
cnn1.add(Flatten())
cnn1.add(Dense(256, activation="relu"))
cnn1.add(Dense(4, activation="relu"))

cnn1.compile(optimizer='sgd', loss='mse', metrics=['accuracy', MeanIoU(num_classes=1)])
print("Compiled!")
#cnn1.summary()

print("Fitting!")
cnn1.fit(x = np.array(x_train), y = y_train, epochs = 10, batch_size = 20)


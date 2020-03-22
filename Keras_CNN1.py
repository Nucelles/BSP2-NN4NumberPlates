import keras

from tensorflow.keras.metrics import MeanIoU

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



# PREPARE THE DATASET FOR TRAINING

path = "/DATA/Training"







# PREPARE THE CNN1 FOR TRAINING
# CNN 1 - Detecting the Lincense Plate
cnn1 = Sequential(name = "CNN1")
cnn1.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(64, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(128, (3, 3), activation="relu"))
cnn1.add(Flatten())
cnn1.add(Dense(256, activation="relu"))
cnn1.add(Dense(4, activation="relu"))

cnn1.compile(optimizer='sgd', loss='mse', metrics=['accuracy', MeanIoU(num_classes=1)])
cnn1.summary()

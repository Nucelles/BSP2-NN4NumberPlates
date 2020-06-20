
from tools import prepareDataset, dice_loss, focal_loss, showModelGraphs
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation

# Activity B
def makeCNN1(imageSize):
    cnn1 = Sequential(name="CNN1")
    cnn1.add(Conv2D(32, (3, 3), input_shape=(imageSize[1], imageSize[0], imageSize[2]), activation="relu"))
    cnn1.add(Conv2D(32, (3, 3), activation="relu"))
    cnn1.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    cnn1.add(Conv2D(64, (3, 3), activation="relu"))
    cnn1.add(Conv2D(64, (3, 3), activation="relu"))
    cnn1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    cnn1.add(Conv2D(128, (3, 3), activation="relu"))
    cnn1.add(Conv2D(128, (3, 3), activation="relu"))
    cnn1.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    cnn1.add(Flatten())

    cnn1.add(Dense(100, activation="relu"))
    cnn1.add(Dropout(0.25))

    cnn1.add(Dense(50, activation="relu"))
    cnn1.add(Dropout(0.25))

    cnn1.add(Dense(4, activation="relu"))

    print("Compiled!")
    cnn1.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
    return cnn1


# PREPARE THE DATASET FOR TRAINING
trainingImageSize = [512, 288, 3]
path = "D:/BiCS/BSP S2/Project/DATA/training"

x_train, y_train, x_val, y_val = prepareDataset(path, 2500, trainingImageSize, 2700, split=True)

cnn1 = makeCNN1(trainingImageSize)

# Activity C
print("Training!")
history = cnn1.fit(x = x_train, y = y_train, epochs = 15, batch_size = 10, validation_data=(x_val, y_val))
print("Finished tranining.")


print("Saving!")
name = (datetime.now().strftime("model_%d-%m-%Y_%I-%M-%S_%p"))
cnn1.save("{}.h5".format(name))
print("Model saved as {}".format(name))

# Activity D
showModelGraphs(history)

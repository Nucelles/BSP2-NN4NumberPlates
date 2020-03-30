from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



trainingImageSize = [30, 15, 3]


cnn2 = Sequential(name ="CNN1")
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
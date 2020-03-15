import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# CNN 1 - Detecting the Lincense Plate
cnn1.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(64, (3, 3), activation="relu"))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Conv2D(128, (3, 3), activation="relu"))
cnn1.add(Flatten())
cnn1.add(Dense(256, activation="relu"))
cnn1.add(Dense(1, activation="softmax")) # WHAT IS THE OUPUT? (shouldn't use softmax)

cnn1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn1.summary()

# CNN 2 - Detecting the Characters
cnn2.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation="relu"))
cnn2.add(MaxPooling2D(pool_size=(2, 2)))
cnn2.add(Conv2D(128, (3, 3), activation="relu"))
cnn2.add(MaxPooling2D(pool_size=(2, 2)))
cnn2.add(Conv2D(256, (3, 3), activation="relu"))
cnn2.add(Flatten())
cnn2.add(Dense(512, activation="relu"))
cnn2.add(Dense(1, activation="softmax")) # WHAT IS THE OUPUT? (shouldn't use softmax)

cnn2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn2.summary()

# CNN 3 - Recognizing the Character
# NOTE: This final NN doesn't really need to be a CNN, can be done with a feedforward

cnn3 = keras.Sequential()

cnn3.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
cnn3.add(MaxPooling2D(pool_size=(2, 2)))
cnn3.add(Conv2D(64, (3, 3), activation="relu"))
cnn3.add(MaxPooling2D(pool_size=(2, 2)))
cnn3.add(Conv2D(128, (3, 3), activation="relu"))
cnn3.add(Flatten())
cnn3.add(Dense(256, activation="relu"))
cnn3.add(Dense(35, activation="softmax"))

cnn3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn3.summary()

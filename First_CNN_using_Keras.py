from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# downloads the MNIST dataset and seperates it into train and test sets
(x_train, y_train), (x_text, y_test) = mnist.load_data()

# use matplotlib to plot the first image in the training set
def showImage():
    plt.imshow(x_train[0])


# PREPROCESSING THE DATA

# the images need to be reshaped
# the images are also in grayscale shown by the "1"
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# one-hot encode target column
# the to_cateogrical is a keras function that one hot encodes data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# CREATING THE MODEL
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# defines the model as a sequential, this means we can use add() method
# to add another layer to the end of the model
model = Sequential()

# first we add 2 convolutional layers, with the first one defining the input size
# the input size is the same as the x_train and x_test shape, (28x28x1) = (wxhxgray)
# -> 64/32, defines the number of nodes in the layer
# -> kernel_size=3, is the size of the filter matrix
# -> activation="relu", the activation function that was used is ReLU
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape = (28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
# Flatten is a connector for the Conv2D layer and the Dense layer
model.add(Flatten())
# We then add the last dense layer as an output layer using softmax
# We have 10 nodes for this layer because there are 10 possible digits, 0-9
model.add(Dense(10, activation='softmax'))


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

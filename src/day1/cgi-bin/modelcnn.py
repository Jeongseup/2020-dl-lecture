"""
Define Convolutional Nerual Network model for MNIST input
"""
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D


class CNN_seq_class(models.Sequential):
    def __init__(self, Nout):
        super().__init__()

        self.add(Conv2D(64, kernel_size=(3, 3), input_shape=(1, 28, 28), data_format='channels_first', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))

        self.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))

        self.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))

        self.add(Flatten())
        self.add(Dense(127, activation='relu'))
        self.add(Dense(Nout, activation='softmax'))
        # Compile model
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


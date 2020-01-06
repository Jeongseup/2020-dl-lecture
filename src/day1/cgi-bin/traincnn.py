import modelcnn
# Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
import numpy as np
from keras.utils import np_utils  # to_categorical
from tensorflow.keras import datasets  # mnist

(X, Y), (testX, testY) = datasets.mnist.load_data()

# one-hot convert
Y = np_utils.to_categorical(Y)

L, W, H = X.shape
X = X.reshape(X.shape[0], 1, W, H)
X = X.astype(np.float64)
X = X / 255.0

number_of_class = 10
Nout = number_of_class

model = modelcnn.CNN_seq_class(Nout)
model.fit(X, Y, epochs=15, batch_size=100, validation_split=0.2)

# Save trained model
model.save_weights("models/modelcnn.tfl")
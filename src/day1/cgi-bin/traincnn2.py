import modelcnn2
import numpy as np
from keras.utils import np_utils  # to_categorical
# Data loading and preprocessing
from tensorflow.keras.datasets import mnist  # mnist

(X, Y), (testX, testY) = mnist.load_data()

# one-hot convert
Y = np_utils.to_categorical(Y)

L, W, H = X.shape
X = X.reshape(X.shape[0], 1, W, H)
X = X.astype(np.float64)
X = X / 255.0

# Train the model
number_of_class = 10
Nout = number_of_class

model = modelcnn2.CNN2_seq_class(Nout)
model.fit(X, Y, epochs=20, batch_size=128, validation_split=0.2)

# Save trained model
model.save_weights("models/modelcnn2.tfl")
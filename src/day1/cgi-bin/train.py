import modelann
import numpy as np
from keras.utils import np_utils  # to_categorical
from tensorflow.keras import datasets  # mnist

(X, Y), (testX, testY) = datasets.mnist.load_data()

# one-hot convert
Y = np_utils.to_categorical(Y)

L, W, H = X.shape
X = X.reshape(-1, W * H)
X = X.astype(np.float64)
X = X / 255.0

Nin = 784
Nh = 256
number_of_class = 10
Nout = number_of_class

model = modelann.ANN_seq_class(Nin, Nh, Nout)
model.fit(X, Y, epochs=20, batch_size=100, validation_split=0.2)

# Save trained model
model.save_weights("models/modelann.tfl")
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import numpy as np

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))

model.summary()

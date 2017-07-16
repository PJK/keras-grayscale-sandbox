import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import tqdm

from data_generator import input_pairs

input_shape = (None, None, 3)

model = Sequential()
model.add(Conv2D(16, kernel_size=(1, 1), use_bias=True,
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(1, kernel_size=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['accuracy'])


def generator():
    while True:
        yield input_pairs(128)

model.fit_generator(generator(), samples_per_epoch=5000)

model.save('grayscale.h5')
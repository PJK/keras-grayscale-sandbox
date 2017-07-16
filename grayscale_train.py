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


for _ in tqdm.trange(100):
    x, y = input_pairs(1000)
    tqdm.tqdm.write("Model loss %f, %f" % tuple(model.train_on_batch(x, y)))

model.save('grayscale.h5')
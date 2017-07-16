import numpy as np
from skimage import color
import matplotlib.pyplot as plt

def input_pairs(count, size = 64):
    x = np.random.uniform(low=0, high=1, size=(count, 64, 64, 3))
    y = np.reshape(np.array([color.rgb2gray(x[i, :, :]) for i in range(count)]), (count, size, size, 1))
    return [x, y]

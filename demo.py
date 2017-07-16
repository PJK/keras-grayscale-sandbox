import sys
import numpy as np
from PIL import Image

def _show(image):
    Image.fromarray((image * 255).astype(np.uint8)).show()

input = np.array(Image.open(sys.argv[1])) / 255.0


from keras.models import load_model
model = load_model('grayscale.h5')
output = model.predict(np.expand_dims(input, axis=0))

Image.fromarray((output[0, :, :, 0] * 255).astype(np.uint8)).save('output.png')


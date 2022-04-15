from tensorflow import keras
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

model = keras.models.load_model('C:\Zaid\College\Federated Learning')

def load_image(filename):
	# load the image
	img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

img = load_image('img_3.jpg')
predict_value = model.predict(img)
digit = np.argmax(predict_value)
print(digit) 
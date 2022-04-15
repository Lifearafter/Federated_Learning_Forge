import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import tensorflow as tf
#importing image and predicting
from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image

(x_train,y_train),(x_test,y_test) = mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#resize using reshape, normalize to display pixel values from 0-1 instead of 0-255
#Here reshape is used and our flatten layer is commented out as reshape does the job of creating a 1D linear vector
#x_train = np.reshape(x_train, [-1,784])
x_train = x_train.astype('float32')/255 #normalize data, show pixel values from 0-1
#x_test = np.reshape(x_test, [-1,784]) #do reshape and normalize for test data too
x_test = x_test.astype('float32')/255

batch_size = 128
hidden_units = 256
dropout =.45

model = Sequential()
model.add(Flatten(input_shape=(28,28))) #flattening layer creates 1D linear vector
model.add(Dense(hidden_units,input_dim=784,activation ='relu'))
model.add(Dropout(dropout)) # use dropout layers to prevent overfitting
model.add(Dense(hidden_units,activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(10,activation='softmax'))


#model.summary() #print out model

model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics= ['accuracy'])
m=model.fit(x_train,y_train, epochs=5, batch_size= batch_size,validation_data=(x_test,y_test))

#model.save('C:\Zaid\College\Federated Learning')

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
 
# load an image and predict the class
def run_example():
	# load the image
 
	img = load_image('img_4.jpg')
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit) 


#training_accuracy = m.history['accuracy']
#test_accuracy = m.history['val_accuracy']
#epoch_count = range(1, len(training_accuracy)+1)

#plt.plot(epoch_count,training_accuracy,'r--')
#plt.plot(epoch_count,test_accuracy,'b-')
#plt.legend(['Training Accuracy', 'Test Accuracy'])
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.title('Accuracy vs Epoch')
#plt.show()

run_example()

# example of loading the mnist dataset
import random
import shutil
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os as os

# load dataset
(trainX, trainy), (testX, testY) = mnist.load_data()
#(dataSetX, dataSetY) = np.concatenate((trainX, testX)), np.concatenate((trainy, testy))
# summarize loaded dataset
testSetX = []
testSetY = []
print('dataSet: X=%s, y=%s' % (testX.shape, testY.shape))
# show the figure
for i in range(50):
    x = random.randint(0,10000)
    np.append(testSetX, testX[x])
    np.append(testSetY, testY[x])
    im = Image.fromarray(testX[x])
    fileName = str(i) + "_MNISTIMG.png"
    if i == 0:
        try:
            currentDirectory = os.getcwd()
            print(currentDirectory)
            path = os.path.join(currentDirectory, 'MNIST_PNG')
            print(path)
            if(os.path.exists(path)):
                print("hi")
                shutil.rmtree(path)
                
            os.mkdir(path)
            os.chdir(path)
        except OSError:
            print("Error")
    im.save(fileName) 

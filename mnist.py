import random

import torchvision.transforms
from torchvision import datasets as dataset
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os as os

DATA_ROOT = os.getcwd() + "./data/mnist"
print(DATA_ROOT)
# load dataset
#transform = torchvision.transforms.Compose(
    #torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.5,), (0.5,)),
#)
trainset = dataset.MNIST(root = DATA_ROOT + "/data", train = True, download=True, transform=None)
testset = dataset.MNIST(root = DATA_ROOT + "/data", train = False, download=True, transform=None)
print(len(testset))
print(type(testset[1][1]))
# (dataSetX, dataSetY) = np.concatenate((trainX, testX)), np.concatenate((trainy, testy))
# summarize loaded dataset
#testSetX = []
#testSetY = []
testSetX = np.empty((28))
testSetY = np.empty(1)

#print('dataSet: X=%s, y=%s' % (testX.shape, testY.shape))
# show the figure
for i in range(3):
    x = random.randint(0, 10000)
    #im_array = np.array(testset[x][0])
    #print(len(im_array))
    #testSetX.append(testset[x][0])
    #testSetY.append(testset[x][1])
    np.concatenate(testSetX, testset[x][0])
    np.concatenate(testSetY, testset[x][1])
    im = testset[x][0]
    fileName = str(i) + "_MNISTIMG.png"
    if i == 0:
        try:
            currentDirectory = os.getcwd()
            path = os.path.join(currentDirectory, 'MNIST_PNG')
            if (not os.path.exists(path)):
                os.mkdir(path)
            os.chdir(path)
        except OSError:
            print("Error")
    im.show()
    im.save(fileName)
print(len(testSetX))
print(len(testSetY))
print("DONE!")

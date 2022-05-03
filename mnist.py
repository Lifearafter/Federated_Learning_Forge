import random
import torchvision
import torchvision.transforms
from torchvision import datasets as dataset
import numpy as np
from PIL import Image
import os as os

DATA_ROOT = os.getcwd() + "/data/mnist"
print(DATA_ROOT)
trainset = dataset.MNIST(root = DATA_ROOT + "/data", train = True, download=True, transform=None)
testset = dataset.MNIST(root = DATA_ROOT + "/data", train = False, download=True, transform=None)

'''
trainset = FastMNIST('data/MNIST', train = True, download=True )
    testset = FastMNIST('data/MNIST', train = False, download=True )
    return trainset, testset
'''
path = os.path.join(os.getcwd(), 'MNIST_PNG')
if (not os.path.exists(path)):
    os.mkdir(path)
os.chdir(path)
f = open('numbers.txt', 'w')
for i in range(51):
    x = random.randint(0, 10000)
    im = testset[x][0]
    im_value = testset[x][1]
    fileName = str(i) + "_MNISTIMG.png"
    im.save(fileName)
    f.write(str(im_value) + "\n")
print("Randomizing Images DONE!")

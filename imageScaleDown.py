from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np

loopBreak = False
while loopBreak is not True:
    fileName = input()
    try:
        img = Image.open(fileName)
        loopBreak = True
    except:
        print("input for filename was incorrect")
    


img = img.resize((28, 28))
image = np.invert(img)
img = Image.fromarray(image)
img = img.convert('L')

img.save('img_4.jpg')


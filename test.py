import os

import cv2
import imageio
import numpy as np

a = np.array([1,2,3,4,1,1,11,2,3,4,5,6])
b = np.where(a == 1)
print(b)
x, y = np.ogrid[:3, :4]
c = np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
print(c)

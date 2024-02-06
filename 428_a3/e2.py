import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = plt.imread('football_field.jpg')
plt.imshow(img)
corners = plt.ginput(4)
# plt.clf()
print(corners)
for i in range(len(corners)):
    corners[i] += (1,)
# print(corners)


diag1 = np.cross(corners[0],corners[2])
diag1 = diag1 / diag1[2]
diag2 = np.cross(corners[1],corners[3])
diag2 = diag2 / diag2[2]
print(f'diagonal lines are: {diag1,diag2}')
midpoint = np.cross(diag1,diag2)
midpoint = midpoint / midpoint[2]
print(f'midpoint: {midpoint}')
plt.scatter(midpoint[0],midpoint[1],c='yellow', s=40) # here
plt.show()

parallel1 = np.cross(corners[0],corners[1])
parallel2 = np.cross(corners[2],corners[3])
vanishing_point = np.cross(parallel1,parallel2)
vanishing_point = vanishing_point/vanishing_point[2]
print(f'vanishing point is {vanishing_point}')
midline = np.cross(midpoint, vanishing_point)
midline = midline/midline[2]
print(f'midline is {midline}')
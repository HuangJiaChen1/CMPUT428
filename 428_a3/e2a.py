import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = plt.imread('football_field.jpg')
plt.imshow(img)
corners = plt.ginput(4)
plt.clf()
plt.close()
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
img_copy = img.copy()
cv2.circle(img_copy,(int(midpoint[0]),int(midpoint[1])),5,(255,0,0),2)
plt.imshow(img_copy)
plt.show()
plt.clf()

parallel1 = np.cross(corners[0],corners[1])
parallel2 = np.cross(corners[2],corners[3])
vanishing_point = np.cross(parallel1,parallel2)
vanishing_point = vanishing_point/vanishing_point[2]
print(f'vanishing point is {vanishing_point}')
midline = np.cross(midpoint, vanishing_point)
midline = midline/midline[2]
print(f'midline is {midline}')
a,b,c = midline
x1, x2 = 0, img_copy.shape[1]  # x values span the width of the image
y1 = int((-a/b) * x1 - (c/b))
y2 = int((-a/b) * x2 - (c/b))
cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
plt.imshow(img_copy)
plt.show()
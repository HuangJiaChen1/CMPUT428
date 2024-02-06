import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

init_point = np.float32([10,10,10,1])
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(init_point[0],init_point[1],init_point[2])


translate_mat = np.float32([
    [1,0,0,0],
    [0,1,0,20],
    [0,0,1,0],
    [0,0,0,1]
])
rot_mat_z = np.float32([
    [math.cos(math.radians(30)), -math.sin(math.radians(30)), 0, 0],
    [math.sin(math.radians(30)), math.cos(math.radians(30)), 0, 0],
    [0,0,1,0],
    [0,0,0,1]
])
rot_mat_y = np.float32([
    [math.cos(math.radians(-10)), 0, math.sin(math.radians(-10)), 0],
    [0 ,1, 0, 0],
    [-math.sin(math.radians(-10)), 0, math.cos(math.radians(-10)), 0],
    [0,0,0,1]
])
trans_point = np.dot(translate_mat,init_point)
rot_z_point = np.dot(rot_mat_z, trans_point)
rot_y_point = np.dot(rot_mat_y,rot_z_point)
print(trans_point)
print(rot_z_point)
print(rot_y_point)
ax.scatter(rot_y_point[0],rot_y_point[1],rot_y_point[2])
plt.show()

plt.plot('football_field.jpg')

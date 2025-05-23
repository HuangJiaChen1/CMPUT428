import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

init_point = np.float32([10,10,10,1])
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(init_point[0],init_point[1],init_point[2])

def affine(coord,trans,rot):
    tx,ty,tz = trans
    rx,ry,rz = rot
    translate_mat = np.float32([
        [tx],
        [ty],
        [tz]
    ])
    rot_mat_z = np.float32([
        [math.cos(math.radians(rz)), -math.sin(math.radians(rz)), 0],
        [math.sin(math.radians(rz)), math.cos(math.radians(rz)), 0],
        [0,0,1]
    ])
    rot_mat_y = np.float32([
        [math.cos(math.radians(ry)), 0, math.sin(math.radians(ry))],
        [0 ,1, 0],
        [-math.sin(math.radians(ry)), 0, math.cos(math.radians(ry))]
    ])
    rot_mat_x = np.float32([
        [1,0,0],
        [0, math.cos(math.radians(rx)), -math.sin(math.radians(rx))],
        [0,math.sin(math.radians(rx)), math.cos(math.radians(rx))]
    ])
    R = np.dot(rot_mat_y,np.dot(rot_mat_z,rot_mat_x))
    trans_coord = np.dot(R,np.add(coord.reshape(-1,1),translate_mat))
    # R[:3,-1] = [tx,ty,tz]
    # print(R)
    T = np.zeros((3,4))
    T[:3,:3] = R
    T[:3, -1] = translate_mat.squeeze()
    return  trans_coord,T

def homography(coord,trans,rot):
    tx,ty,tz = trans
    rx,ry,rz = rot
    translate_mat = np.float32([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]
    ])
    rot_mat_z = np.float32([
        [math.cos(math.radians(rz)), -math.sin(math.radians(rz)), 0, 0],
        [math.sin(math.radians(rz)), math.cos(math.radians(rz)), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    rot_mat_y = np.float32([
        [math.cos(math.radians(ry)), 0, math.sin(math.radians(ry)), 0],
        [0 ,1, 0, 0],
        [-math.sin(math.radians(ry)), 0, math.cos(math.radians(ry)), 0],
        [0,0,0,1]
    ])
    rot_mat_x = np.float32([
        [1,0,0,0],
        [0, math.cos(math.radians(rx)), -math.sin(math.radians(rx)), 0],
        [0,math.sin(math.radians(rx)), math.cos(math.radians(rx)), 0],
        [0, 0, 0, 1]
    ])
    R = np.dot(rot_mat_y,np.dot(rot_mat_z,rot_mat_x))
    # T = np.dot(R,translate_mat)
    trans_coord = np.dot(R,np.dot(translate_mat,coord.reshape(-1,1)))
    R[:3,-1] = [tx,ty,tz]
    # print(R)

    return  trans_coord,R

homo_point,T = homography(init_point, (0, 20, 0), (0, -10, 30))
homo_point = homo_point / homo_point[3]
print(homo_point)
print(T)
coord_sys, T = homography(init_point,(0,-20,0),(0,10,-30))
coord_sys = coord_sys / coord_sys[3]
print(coord_sys)
print(T)
ax.scatter(homo_point[0], homo_point[1], homo_point[2])
ax.scatter(coord_sys[0],coord_sys[1],coord_sys[2])
plt.show()

init_point_affine = np.float32([10,10,10])
affine_point,T = affine(init_point_affine, (0, 20, 0), (0, -10, 30))
print(affine_point)
print(T)
coord_sys_affine, T = affine(init_point_affine, (0, -20, 0), (0, 10, -30))
print(coord_sys_affine)
print(T)
ax.scatter(affine_point[0], affine_point[1], affine_point[2])
ax.scatter(coord_sys_affine[0], coord_sys_affine[1], coord_sys_affine[2])
plt.show()
# plt.plot('football_field.jpg')

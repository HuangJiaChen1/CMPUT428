import math

import numpy as np
from matplotlib import pyplot as plt

ortho_mat = np.float32([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1]
])
p = np.float32([
    [0.5,0,0],
    [0,0.5,0],
    [0,0,1]
])
ijk = np.float32([
    [1,0,0,10],
    [0,1,0,10],
    [0,0,1,10],
])
perspective = np.dot(p,ijk)
# print(perspective)
rot = np.float32([
    [math.cos(math.radians(45)), math.sin(math.radians(45)),0,0],
    [0,1, 0,0],
    [-math.sin(math.radians(45)), math.cos(math.radians(45)),1,0],
    [0,0,0,1]
])
# Rectangle
def rectangle():
    x_edges = np.concatenate([np.random.rand(25), np.ones(25), np.random.rand(25), np.zeros(25)])
    y_edges = np.concatenate([np.zeros(25), np.random.rand(25), np.ones(25), np.random.rand(25)])
    z_edges = -np.ones(100)
    rectangle = np.vstack([x_edges, y_edges, z_edges,np.ones(100)])

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rectangle[0], rectangle[1], rectangle[2])
    plt.show()
    rectangle_ortho = np.dot(ortho_mat, rectangle)
    print(rectangle_ortho)
    rectangle_ortho = rectangle_ortho[:2, :] / rectangle_ortho[2, :]
    plt.scatter(rectangle_ortho[0],rectangle_ortho[1])
    plt.show()
    rectangle_pers = np.dot(perspective,rectangle)
    result = rectangle_pers[:2, :] / rectangle_pers[2, :]
    plt.scatter(result[0],result[1])
    plt.show()
    r = np.dot(rot,rectangle)
    rp = np.dot(perspective,r)
    result = rp[:2, :] / rp[2, :]
    plt.scatter(result[0],result[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()



# Circle
def circle():
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)
    z = np.zeros(100)
    circle = np.vstack([x, y, z,np.ones((1,100))])
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(circle[0], circle[1], circle[2])
    plt.show()
    circle_ortho = np.dot(ortho_mat, circle)
    plt.scatter(circle_ortho[0],circle_ortho[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    circle_pers = np.dot(perspective,circle)
    result = circle_pers[:2, :] / circle_pers[2, :]
    plt.scatter(result[0],result[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    r = np.dot(rot,circle)
    rp = np.dot(perspective,r)
    result = rp[:2, :] / rp[2, :]
    plt.scatter(result[0], result[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

def cube():
    points = []
    for edge in [(0, 1), (1, 1), (1, 0), (0, 0), (0, 0)]:
        x = np.linspace(0, 1, 100)
        y = np.ones(100) * edge[0]
        z = np.ones(100) * edge[1]
        points.append(np.vstack([x, y, z]))
        points.append(np.vstack([y, x, z]))
        points.append(np.vstack([y, z, x]))
    cube = np.concatenate(points, axis=1)
    print(cube.shape)
    cube = np.vstack((cube,np.ones((1,1500))))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cube[0], cube[1], cube[2], s=1)
    plt.show()
    cube_ortho = np.dot(ortho_mat, cube)
    plt.scatter(cube_ortho[0],cube_ortho[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    cube_perspective = np.dot(perspective,cube)
    print(cube_perspective)
    cube_perspective = cube_perspective[:2, :] / cube_perspective[2, :]
    plt.scatter(cube_perspective[0],cube_perspective[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    r = np.dot(rot,cube)
    rp = np.dot(perspective,r)
    result = rp[:2, :] / rp[2, :]
    plt.scatter(result[0],result[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()




def line():
    x = np.random.rand(100,1)
    # print(x.shape)
    y = np.zeros((100,1))
    z = np.zeros((100,1))
    line = np.squeeze(np.float32([
        x.T,
        y.T,
        z.T,
        np.ones((100,1)).T
    ]))
    # print(line.shape)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(line[0],line[1],line[2])
    plt.show()
    line_ortho = np.dot(ortho_mat, line)
    plt.scatter(line_ortho[0],line_ortho[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    line_pers = np.dot(perspective,line)
    result = line_pers[:2, :] / line_pers[2, :]
    plt.scatter(result[0],result[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    r = np.dot(rot,line)
    rp = np.dot(perspective,r)
    result = rp[:2, :] / rp[2, :]
    plt.scatter(result[0], result[1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

cube()
line()
rectangle()
circle()

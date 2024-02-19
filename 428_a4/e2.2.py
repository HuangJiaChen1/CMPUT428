import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
p = np.float32([
    [600,0,0],
    [0,600,0],
    [0,0,1]
])
ijk = np.float32([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,10],
])
perspective = np.dot(p,ijk) #3x4
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
vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
for i, vertex in enumerate(vertices):
    ax.scatter(vertex[0], vertex[1], vertex[2], c=colors[i], s=50)  # Larger size for visibility
vertices = np.hstack([vertices,np.ones((8,1))])
# print(vertices.T)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()
xl = []
yl = []
x_diffs = []
y_diffs = []
prev_vertex_perspective = None

for i in range(10):
    cube_perspective = np.dot(perspective, cube)
    vertex_perspective = np.dot(perspective, vertices.T)
    cube_perspective = cube_perspective[:2, :] / cube_perspective[2, :]
    # print(cube_perspective)
    plt.scatter(cube_perspective[0], cube_perspective[1])
    vertex_perspective = vertex_perspective[:2, :] / vertex_perspective[2, :]
    # print(vertex_perspective)
    for j in range(vertex_perspective.shape[1]):
        plt.scatter(vertex_perspective[0][j], vertex_perspective[1][j], c=colors[j])  # Larger size for visibility
    plt.show()
    if prev_vertex_perspective is not None:
        x_diff = vertex_perspective[0, :] - prev_vertex_perspective[0, :]
        y_diff = vertex_perspective[1, :] - prev_vertex_perspective[1, :]
        x_diffs.append(x_diff)
        y_diffs.append(y_diff)
        xl.append(prev_vertex_perspective[0,:]-ijk[0,-1])
        yl.append(prev_vertex_perspective[1,:]-ijk[1,-1])
    prev_vertex_perspective = vertex_perspective

    ijk[0, -1] += 1
    perspective = np.dot(p, ijk)
x_diffs = np.array(x_diffs)
print(x_diffs)
y_diffs = np.array(y_diffs)
xl = np.array(xl)
yl = np.array(yl)
print(xl)

# fb = np.zeros((9,1))
# fb+=600
# z = fb/x_diffs
# print(z)
# z = np.mean(z,axis=0)
# x = xl*z/fb
# x = np.mean(x,axis=0)
# print(x)
# y = yl*z/fb
# y = np.mean(y,axis=0)
# print(y)
# print(z)


fb = np.zeros((9,1))
fb+=500*8
z,  residuals, rank, s= np.linalg.lstsq(x_diffs,fb,rcond=None)
print(z, residuals, rank, s)
f = np.zeros((8,1))
f += 500
print(np.dot(xl,z))
# x, residuals, rank, s = np.linalg.lstsq(f,np.dot(xl[0],z),rcond=None)
# y = np.linalg.lstsq(f,np.dot(yl[0],z),rcond=None)[0]
x = xl[0]*z.T/f.T
y = yl[0]*z.T/f.T
print(x)
print(y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.squeeze(x),np.squeeze(y),-np.squeeze(z.T))
# plt.xlim([-1,1])
# plt.ylim([-1,1])
plt.show()
import numpy as np
from matplotlib import pyplot as plt

coordinates = np.array([
    [(389, 315), (393, 326), (394, 336), (393, 351), (521, 357), (523, 332), (516, 329), (504, 318)],
    [(365, 315), (368, 326), (369, 335), (368, 350), (497, 356), (499, 331), (492, 327), (480, 317)],
    [(339, 314), (343, 324), (342, 334), (342, 350), (472, 356), (474, 331), (467, 327), (456, 317)],
    [(313, 313), (316, 324), (315, 334), (314, 349), (446, 357), (447, 331), (441, 327), (432, 316)],
    [(284, 312), (285, 323), (284, 333), (282, 347), (415, 356), (417, 331), (410, 326), (403, 316)],
    [(255, 312), (256, 322), (254, 332), (252, 347), (386, 355), (388, 330), (381, 326), (375, 316)],
    [(229, 311), (229, 321), (226, 331), (226, 346), (358, 355), (360, 330), (353, 326), (348, 315)],
    [(203, 310), (202, 320), (200, 330), (198, 345), (329, 354), (332, 329), (325, 325), (323, 314)],
    [(176, 309), (174, 320), (171, 329), (170, 343), (302, 353), (304, 328), (298, 324), (295, 313)],
    [(152, 308), (149, 318), (146, 328), (145, 343), (274, 353), (277, 328), (271, 324), (270, 312)],
    [(123, 307), (121, 317), (115, 327), (115, 341), (242, 352), (244, 326), (238, 323), (238, 312)]
])


coordinates = np.array(coordinates)
x_diffs = np.diff(coordinates[:,:,0], axis=0)
y_diffs = np.diff(coordinates[:,:,1], axis=0)

print(x_diffs, y_diffs)
xl = coordinates[:, :, 0]
yl = coordinates[:, :, 1]
print(xl,yl)

# fb = np.zeros((10,1))
# fb+=500*8
# z,  residuals, rank, s= np.linalg.lstsq(x_diffs,fb,rcond=None)
# print(z, residuals, rank, s)
# f = np.zeros((8,1))
# f += 500
# print(np.dot(xl,z))
# # x, residuals, rank, s = np.linalg.lstsq(f,np.dot(xl[0],z),rcond=None)
# # y = np.linalg.lstsq(f,np.dot(yl[0],z),rcond=None)[0]
# x = xl[0]*z.T/f.T
# y = yl[0]*z.T/f.T
# print(x)
# print(y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.squeeze(x),np.squeeze(y),-np.squeeze(z.T))
# # plt.xlim([-1,1])
# # plt.ylim([-1,1])
# plt.show()

fb = np.zeros((10,1))
fb+=500
f = np.zeros((8,1))
f += 500
z = fb/x_diffs
print(z)
z = np.mean(z,axis=0)
x = xl*z/f.T
x = np.mean(x,axis=0)
print(x)
y = yl*z/f.T
y = np.mean(y,axis=0)
print(y)
print(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.squeeze(x),-np.squeeze(z.T),np.squeeze(y))
# plt.xlim([-1,1])
# plt.ylim([-1,1])
plt.show()

# disparities = np.abs(np.diff(xl, axis=0))
# mean_disparities = np.mean(disparities, axis=0)
# fb = np.zeros((10,1))
# fb+=500*8
# z,  residuals, rank, s= np.linalg.lstsq(x_diffs,fb,rcond=None)
# print(z)
# # Setup matrix A for least squares (Ax = B), where A will just be the reciprocal of the mean disparities
# A = np.ones_like(mean_disparities) / mean_disparities
# B = np.ones_like(mean_disparities) * (500 * 1)  # B is the product f * b, constant for all points
# z= (500 * 1) / mean_disparities
# f = np.zeros((8,1))
# f += 500
# print(z)
# x = xl[0]*z.T/f.T
# y = yl[0]*z.T/f.T
# print(x)
# print(y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.squeeze(x),np.squeeze(y),-np.squeeze(z.T))
# # plt.xlim([-1,1])
# # plt.ylim([-1,1])
# plt.show()

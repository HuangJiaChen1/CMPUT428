import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
def rearrange_data_ndarray(W):
    half = W.shape[0] // 2
    X = W[:half]
    Y = W[half:]
    rearranged = np.empty(W.shape, dtype=W.dtype)
    rearranged[0::2], rearranged[1::2] = X, Y
    return rearranged

# Load pts and imgs
data = loadmat("HouseTallBasler64.mat")
n_frames = data["NrFrames"].item()
# print(data)
W = data["W"]
# print(W)
# print("  ")
# print(W)
imgs = data["mexVims"]
# print("W:", W.shape)

# Plot
# for frame_i in range(n_frames):
    # img = cv2.cvtColor(imgs[:, :, frame_i], cv2.COLOR_BAYER_BG2RGB)
    # img = cv2.resize(img, (240, 320))
    #
    # y = W[frame_i, :]
    # x = W[n_frames + frame_i, :]
    #
    # plt.imshow(img)
    # plt.scatter(x, y)
    # plt.pause(0.1)
    # plt.clf()
print(W[:2])
W = rearrange_data_ndarray(W)
mean_W = np.mean(W,axis=1)
for i in range(W.shape[1]):
    W[:,i] -= mean_W
U,D,VT = np.linalg.svd(W)
M = np.dot(U[:,:3],np.diag(D[:3]))
print(M)
print(VT.shape)
coord = VT[:3,:]
z= coord[0,:]
y = coord[1,:]
x = coord[2,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


data2 = loadmat("affrec1.mat")
W = data2["W"].astype("float64")
W = rearrange_data_ndarray(W)
mean_W = np.mean(W,axis=1)
for i in range(W.shape[1]):
    W[:,i] -= mean_W
U,D,VT = np.linalg.svd(W)
M = np.dot(U[:,:3],np.diag(D[:3]))
print(M)
print(VT.shape)
coord = VT[:3,:]
z= coord[0,:]
y = coord[1,:]
x = coord[2,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

data3 = loadmat("affrec1.mat")
W = data3["W"].astype("float64")
W = rearrange_data_ndarray(W)
mean_W = np.mean(W,axis=1)
for i in range(W.shape[1]):
    W[:,i] -= mean_W
U,D,VT = np.linalg.svd(W)
M = np.dot(U[:,:3],np.diag(D[:3]))
print(M)
print(VT.shape)
coord = VT[:3,:]
z= coord[0,:]
y = coord[1,:]
x = coord[2,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
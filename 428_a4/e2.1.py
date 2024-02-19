import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

XL = []
YL = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        XL.append(x-image.shape[0]/2)
        YL.append(y-image.shape[1]/2)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Image', image)
cap = cv2.VideoCapture(0)

ret, image = cap.read()

cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)
cap.release()
while True:
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()

XR = []
YR = []
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        XR.append(x-image.shape[0])
        YR.append(y-image.shape[1])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Image', image)
cap = cv2.VideoCapture(0)

ret, image = cap.read()

cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event2)
cap.release()
while True:
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()

focal_length = 500
# print(XL)
# print(YL)
d = 5
plot_point_x = []
plot_point_y = []
plot_point_z = []
for i in range(len(XL)):
    xl = XL[i]
    xr = XR[i]
    yl = YL[i]
    Z = d*focal_length/(xl-xr)
    # print(Z)
    x = xl*Z/focal_length
    y = -yl*Z/focal_length
    plot_point_x.append(x)
    plot_point_y.append(Z)
    plot_point_z.append(y)
print(plot_point_x,
    plot_point_y,
    plot_point_z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plot_point_x, plot_point_y, plot_point_z)
plt.show()
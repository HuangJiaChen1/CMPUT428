import cv2
XL = []
YL = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        XL.append(x)
        YL.append(y)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Image', image)
cap = cv2.VideoCapture(0)

ret, image = cap.read()

cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)

# Release the VideoCapture object
cap.release()

# Wait for the 'Enter' key press
while True:
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cv2.destroyAllWindows()

XR = []
YR = []
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        XR.append(x)
        YR.append(y)
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

focal_length = 600
print(XL)
print(YL)
d = 5
plot_point = []
for i in range(len(XL)):
    xl = XL[i]
    xr = XR[i]
    yl = YL[i]
    Z = d*focal_length/(xl-xr)
    print(Z)
    x = xl*Z/focal_length
    y = yl*Z/focal_length
    plot_point.append((x,y))

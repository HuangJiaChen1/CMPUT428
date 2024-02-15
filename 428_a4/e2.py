import cv2
import numpy as np
import math
points = []
distance = 0.0
def click_event(event, x, y, flags, param):
    global distance
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Mark the clicked point and show it on the image
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        if len(points) >= 2:
            cv2.line(frame, points[-2], points[-1], (255, 0, 0), 2)
        cv2.imshow('Captured Image', frame)

        # When two points are selected, calculate and print the distance
        if len(points) == 2:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            distance += math.sqrt(dx ** 2 + dy ** 2)
cap = cv2.VideoCapture(0)

_, _ = cap.read()
_, _ = cap.read()
ret, frame = cap.read()

cv2.imshow('Captured Image', frame)
cv2.setMouseCallback('Captured Image', click_event)
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

print(distance)

ratio = 0.5
focal_length = distance/ratio
print(focal_length)
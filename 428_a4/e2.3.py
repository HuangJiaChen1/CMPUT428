import cv2
import sys

import numpy as np


def select_point(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.namedWindow("Frame")

points = []
cv2.setMouseCallback("Frame", select_point)

while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

trackers = cv2.MultiTracker_create()
for point in points:
    trackers.add(cv2.TrackerCSRT_create(), frame, (point[0] - 10, point[1] - 10, 20, 20))
saved_frames = 0
coordinates = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, boxes = trackers.update(frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # Save the current frame
        img_name = f"frame_{saved_frames}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")

        # Collect and print coordinates
        current_coords = [(int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)) for box in boxes]
        coordinates.append(current_coords)
        print(f"Coordinates for {img_name}: {current_coords}")

        saved_frames += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("All collected coordinates:", coordinates)
#[[(389, 315), (393, 326), (394, 336), (393, 351), (521, 357), (523, 332), (516, 329), (504, 318)], 
# [(365, 315), (368, 326), (369, 335), (368, 350), (497, 356), (499, 331), (492, 327), (480, 317)], 
# [(339, 314), (343, 324), (342, 334), (342, 350), (472, 356), (474, 331), (467, 327), (456, 317)], 
# [(313, 313), (316, 324), (315, 334), (314, 349), (446, 357), (447, 331), (441, 327), (432, 316)], 
# [(284, 312), (285, 323), (284, 333), (282, 347), (415, 356), (417, 331), (410, 326), (403, 316)], 
# [(255, 312), (256, 322), (254, 332), (252, 347), (386, 355), (388, 330), (381, 326), (375, 316)], 
# [(229, 311), (229, 321), (226, 331), (226, 346), (358, 355), (360, 330), (353, 326), (348, 315)], 
# [(203, 310), (202, 320), (200, 330), (198, 345), (329, 354), (332, 329), (325, 325), (323, 314)], 
# [(176, 309), (174, 320), (171, 329), (170, 343), (302, 353), (304, 328), (298, 324), (295, 313)], 
# [(152, 308), (149, 318), (146, 328), (145, 343), (274, 353), (277, 328), (271, 324), (270, 312)], 
# [(123, 307), (121, 317), (115, 327), (115, 341), (242, 352), (244, 326), (238, 323), (238, 312)]]



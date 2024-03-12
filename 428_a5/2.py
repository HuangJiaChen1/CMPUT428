import cv2
import numpy as np
import os

# Path to the directory containing frames
frames_path = 'frames'
frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])

# Check if we have enough frames
if len(frame_files) < 120:
    raise ValueError("Not enough frames in the directory.")

# Read the first frame
first_frame_path = os.path.join(frames_path, frame_files[2])
frame = cv2.imread(first_frame_path)
if frame is None:
    raise IOError(f"Failed to read the first frame from {first_frame_path}")

# List to store the points
points = []
trackers = []

def select_point(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
cv2.namedWindow("Frame")

cv2.setMouseCallback("Frame", select_point)

while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

trackers = cv2.MultiTracker_create()
for point in points:
    trackers.add(cv2.TrackerCSRT_create(), frame, (point[0] - 10, point[1] - 10, 20, 20))
coords_list = []

# Track points for the next 120 frames (or as many as we have)
for frame_file in frame_files[:120]:
    frame_path = os.path.join(frames_path, frame_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        break

    frame_coords = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x_center = int(bbox[0] + bbox[2] / 2)
            y_center = int(bbox[1] + bbox[3] / 2)
            frame_coords.append([x_center, y_center])
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

    coords_list.append(frame_coords)
    cv2.imshow("Tracking", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(points)
cv2.destroyAllWindows()
print(coords_list)
coords_array = np.array(coords_list)
W = coords_array.transpose(2, 0, 1).reshape(2, -1).T

# W is your final matrix
print(W.shape)
print(W)

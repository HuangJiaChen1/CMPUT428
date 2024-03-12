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
num_points = 0
def select_point(event, x, y, flags, params):
    global num_points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        num_points += 1
cv2.namedWindow("Frame")

cv2.setMouseCallback("Frame", select_point)

while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
for point in points:
    tracker= cv2.TrackerMIL_create()
    tracker.init(frame, (point[0] - 10, point[1] - 10, 20, 20))
    trackers.append(tracker)
print(trackers)
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
print(coords_array.shape)
W = coords_array.transpose(0,2,1).reshape(240,num_points)
print(W.shape)
print(W)
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